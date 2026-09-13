-- SARAL parser + chunking schema. Apply through a reviewed Supabase migration.
create table if not exists public.documents (
  id text primary key,
  owner_id uuid not null references auth.users(id) on delete cascade,
  content_sha256 text not null check (length(content_sha256) = 64),
  original_filename text not null,
  media_type text not null,
  size_bytes bigint not null check (size_bytes > 0),
  status text not null check (status in ('queued', 'processing', 'ready', 'failed')),
  original_storage_path text not null,
  docling_json_storage_path text,
  parser_options jsonb not null default '{}'::jsonb,
  chunker_options jsonb not null default '{}'::jsonb,
  page_count integer not null default 0,
  figure_count integer not null default 0,
  table_count integer not null default 0,
  formula_count integer not null default 0,
  chunk_count integer not null default 0,
  warnings jsonb not null default '[]'::jsonb,
  errors jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (owner_id, content_sha256)
);

create table if not exists public.processing_jobs (
  id uuid primary key,
  document_id text not null references public.documents(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  job_type text not null check (job_type = 'parse_and_chunk'),
  status text not null check (status in ('queued', 'processing', 'ready', 'failed')),
  configuration jsonb not null default '{}'::jsonb,
  error_code text,
  error_message text,
  created_at timestamptz not null default now(),
  started_at timestamptz,
  completed_at timestamptz
);

create table if not exists public.document_assets (
  id text primary key,
  document_id text not null references public.documents(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  asset_type text not null check (asset_type in ('figure', 'table', 'page')),
  storage_path text not null unique,
  media_type text not null,
  page_number integer,
  source_ref text,
  bounding_box jsonb,
  caption jsonb not null default '[]'::jsonb,
  generated_description text,
  created_at timestamptz not null default now()
);

create table if not exists public.document_validations (
  document_id text not null references public.documents(id) on delete cascade,
  job_id uuid not null references public.processing_jobs(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  status text not null,
  warnings jsonb not null default '[]'::jsonb,
  failures jsonb not null default '[]'::jsonb,
  review_items jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now(),
  primary key (document_id, job_id)
);

create table if not exists public.document_chunks (
  id text primary key,
  document_id text not null references public.documents(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  job_id uuid not null references public.processing_jobs(id) on delete cascade,
  chunk_index integer not null check (chunk_index >= 0),
  text text not null check (length(text) > 0),
  contextualized_text text not null check (length(contextualized_text) > 0),
  headings jsonb not null default '[]'::jsonb,
  captions jsonb not null default '[]'::jsonb,
  source_refs jsonb not null,
  page_numbers jsonb not null default '[]'::jsonb,
  provenance jsonb not null default '[]'::jsonb,
  asset_ids jsonb not null default '[]'::jsonb,
  content_types jsonb not null default '[]'::jsonb,
  token_count integer not null check (token_count > 0),
  chunker_version text not null,
  created_at timestamptz not null default now(),
  unique (document_id, chunk_index)
);

create index if not exists processing_jobs_document_id_idx on public.processing_jobs (document_id);
create index if not exists processing_jobs_owner_status_idx on public.processing_jobs (owner_id, status);
create index if not exists document_assets_document_id_idx on public.document_assets (document_id);
create index if not exists document_assets_owner_document_idx on public.document_assets (owner_id, document_id);
create index if not exists document_validations_job_id_idx on public.document_validations (job_id);
create index if not exists document_chunks_document_order_idx on public.document_chunks (document_id, chunk_index);
create index if not exists document_chunks_job_id_idx on public.document_chunks (job_id);

alter table public.documents enable row level security;
alter table public.processing_jobs enable row level security;
alter table public.document_assets enable row level security;
alter table public.document_validations enable row level security;
alter table public.document_chunks enable row level security;

create policy "owners read documents" on public.documents for select to authenticated
using ((select auth.uid()) = owner_id);
create policy "owners read jobs" on public.processing_jobs for select to authenticated
using ((select auth.uid()) = owner_id);
create policy "owners read assets" on public.document_assets for select to authenticated
using ((select auth.uid()) = owner_id);
create policy "owners read validations" on public.document_validations for select to authenticated
using ((select auth.uid()) = owner_id);
create policy "owners read chunks" on public.document_chunks for select to authenticated
using ((select auth.uid()) = owner_id);

-- The bucket is private. Only the Python API/worker uses the service-role key.
insert into storage.buckets (id, name, public)
values ('saral-documents', 'saral-documents', false)
on conflict (id) do update set public = false;

create policy "owners read their storage prefix" on storage.objects for select to authenticated
using (bucket_id = 'saral-documents' and (storage.foldername(name))[1] = (select auth.uid())::text);
