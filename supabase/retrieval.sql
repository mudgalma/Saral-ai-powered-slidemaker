-- SARAL retrieval index migration. Apply after supabase/schema.sql.
-- The API and workers call these functions with a verified owner ID using the
-- service-role connection; browser roles have no execute privilege.

create extension if not exists vector with schema extensions;

alter table public.documents
  add column if not exists embedding_status text not null default 'not_started'
    check (embedding_status in ('not_started', 'queued', 'processing', 'ready', 'failed')),
  add column if not exists embedding_model text,
  add column if not exists embedding_error text;

alter table public.processing_jobs
  drop constraint if exists processing_jobs_job_type_check;

alter table public.processing_jobs
  add constraint processing_jobs_job_type_check
  check (job_type in ('parse_and_chunk', 'embed_document'));

alter table public.document_chunks
  add column if not exists embedding extensions.vector(1536),
  add column if not exists embedding_model text,
  add column if not exists embedded_at timestamptz,
  add column if not exists search_vector tsvector
    generated always as (to_tsvector('english'::regconfig, contextualized_text)) stored;

create index if not exists document_chunks_owner_document_idx
  on public.document_chunks (owner_id, document_id);

create index if not exists document_chunks_search_vector_idx
  on public.document_chunks using gin (search_vector);

create index if not exists document_chunks_embedding_hnsw_idx
  on public.document_chunks
  using hnsw (embedding extensions.vector_cosine_ops)
  where embedding is not null;

create or replace function public.write_document_chunk_embeddings(
  p_owner_id uuid,
  p_document_id text,
  p_embedding_model text,
  p_embeddings jsonb
)
returns integer
language plpgsql
set search_path = public, extensions
as $$
declare
  updated_count integer;
begin
  update public.document_chunks as chunk
  set
    embedding = (item.embedding)::extensions.vector,
    embedding_model = p_embedding_model,
    embedded_at = now()
  from jsonb_to_recordset(p_embeddings) as item(chunk_id text, embedding text)
  where chunk.id = item.chunk_id
    and chunk.owner_id = p_owner_id
    and chunk.document_id = p_document_id;

  get diagnostics updated_count = row_count;
  return updated_count;
end;
$$;

create or replace function public.search_document_chunks_dense(
  p_owner_id uuid,
  p_document_id text,
  p_query_embedding extensions.vector(1536),
  p_match_count integer default 20
)
returns table (
  chunk_id text,
  document_id text,
  chunk_index integer,
  text text,
  contextualized_text text,
  headings jsonb,
  captions jsonb,
  source_refs jsonb,
  page_numbers jsonb,
  provenance jsonb,
  asset_ids jsonb,
  content_types jsonb,
  score double precision
)
language sql
stable
set search_path = public, extensions
as $$
  select
    chunk.id,
    chunk.document_id,
    chunk.chunk_index,
    chunk.text,
    chunk.contextualized_text,
    chunk.headings,
    chunk.captions,
    chunk.source_refs,
    chunk.page_numbers,
    chunk.provenance,
    chunk.asset_ids,
    chunk.content_types,
    1 - (chunk.embedding <=> p_query_embedding) as score
  from public.document_chunks as chunk
  where chunk.owner_id = p_owner_id
    and chunk.document_id = p_document_id
    and chunk.embedding is not null
  order by chunk.embedding <=> p_query_embedding asc
  limit least(greatest(p_match_count, 1), 20);
$$;

create or replace function public.search_document_chunks_sparse(
  p_owner_id uuid,
  p_document_id text,
  p_query_text text,
  p_match_count integer default 20
)
returns table (
  chunk_id text,
  document_id text,
  chunk_index integer,
  text text,
  contextualized_text text,
  headings jsonb,
  captions jsonb,
  source_refs jsonb,
  page_numbers jsonb,
  provenance jsonb,
  asset_ids jsonb,
  content_types jsonb,
  score real
)
language sql
stable
set search_path = public, extensions
as $$
  with query as (
    select websearch_to_tsquery('english'::regconfig, p_query_text) as value
  )
  select
    chunk.id,
    chunk.document_id,
    chunk.chunk_index,
    chunk.text,
    chunk.contextualized_text,
    chunk.headings,
    chunk.captions,
    chunk.source_refs,
    chunk.page_numbers,
    chunk.provenance,
    chunk.asset_ids,
    chunk.content_types,
    ts_rank_cd(chunk.search_vector, query.value) as score
  from public.document_chunks as chunk
  cross join query
  where chunk.owner_id = p_owner_id
    and chunk.document_id = p_document_id
    and chunk.search_vector @@ query.value
  order by score desc, chunk.chunk_index asc
  limit least(greatest(p_match_count, 1), 20);
$$;

revoke all on function public.search_document_chunks_dense(uuid, text, extensions.vector, integer)
  from public, anon, authenticated;
revoke all on function public.search_document_chunks_sparse(uuid, text, text, integer)
  from public, anon, authenticated;
revoke all on function public.write_document_chunk_embeddings(uuid, text, text, jsonb)
  from public, anon, authenticated;
grant execute on function public.search_document_chunks_dense(uuid, text, extensions.vector, integer)
  to service_role;
grant execute on function public.search_document_chunks_sparse(uuid, text, text, integer)
  to service_role;
grant execute on function public.write_document_chunk_embeddings(uuid, text, text, jsonb)
  to service_role;
