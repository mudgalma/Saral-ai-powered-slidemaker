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

create table if not exists public.conversations (
  id uuid primary key,
  document_id text not null references public.documents(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.conversation_messages (
  id uuid primary key,
  conversation_id uuid not null references public.conversations(id) on delete cascade,
  document_id text not null references public.documents(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null check (length(content) between 1 and 16000),
  created_at timestamptz not null default now()
);

create table if not exists public.artifact_versions (
  id uuid primary key,
  conversation_id uuid not null references public.conversations(id) on delete cascade,
  document_id text not null references public.documents(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  version_number integer not null check (version_number > 0),
  parent_version_id uuid references public.artifact_versions(id) on delete set null,
  artifact jsonb not null,
  delta text check (delta is null or length(delta) <= 8000),
  created_at timestamptz not null default now(),
  unique (conversation_id, version_number)
);

create index if not exists conversations_owner_document_idx on public.conversations (owner_id, document_id);
create index if not exists conversation_messages_conversation_created_idx on public.conversation_messages (conversation_id, created_at desc);
create index if not exists conversation_messages_document_owner_idx on public.conversation_messages (document_id, owner_id);
create index if not exists artifact_versions_conversation_version_idx on public.artifact_versions (conversation_id, version_number desc);
create index if not exists artifact_versions_document_owner_idx on public.artifact_versions (document_id, owner_id);
create index if not exists artifact_versions_parent_version_idx on public.artifact_versions (parent_version_id);

create or replace function public.create_artifact_version(
  p_id uuid,
  p_conversation_id uuid,
  p_document_id text,
  p_owner_id uuid,
  p_artifact jsonb,
  p_parent_version_id uuid,
  p_delta text
) returns public.artifact_versions
language plpgsql
security invoker
set search_path = public
as $$
declare
  next_version integer;
  created_version public.artifact_versions;
begin
  if not exists (
    select 1 from public.conversations
    where id = p_conversation_id and document_id = p_document_id and owner_id = p_owner_id
  ) then
    raise exception 'conversation scope is invalid' using errcode = '42501';
  end if;

  perform pg_advisory_xact_lock(hashtext(p_conversation_id::text));
  select coalesce(max(version_number), 0) + 1 into next_version
  from public.artifact_versions
  where conversation_id = p_conversation_id and document_id = p_document_id and owner_id = p_owner_id;

  insert into public.artifact_versions (
    id, conversation_id, document_id, owner_id, version_number, parent_version_id, artifact, delta
  ) values (
    p_id, p_conversation_id, p_document_id, p_owner_id, next_version,
    p_parent_version_id, p_artifact, p_delta
  ) returning * into created_version;
  return created_version;
end;
$$;

revoke all on function public.create_artifact_version(uuid, uuid, text, uuid, jsonb, uuid, text)
from public, anon, authenticated;
grant execute on function public.create_artifact_version(uuid, uuid, text, uuid, jsonb, uuid, text)
to service_role;

alter table public.documents enable row level security;
alter table public.processing_jobs enable row level security;
alter table public.document_assets enable row level security;
alter table public.document_validations enable row level security;
alter table public.document_chunks enable row level security;
alter table public.conversations enable row level security;
alter table public.conversation_messages enable row level security;
alter table public.artifact_versions enable row level security;

revoke all on table public.conversations from anon, authenticated;
revoke all on table public.conversation_messages from anon, authenticated;
revoke all on table public.artifact_versions from anon, authenticated;
grant select on table public.conversations to authenticated;
grant select on table public.conversation_messages to authenticated;
grant select on table public.artifact_versions to authenticated;

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
create policy "owners read conversations" on public.conversations for select to authenticated
using ((select auth.uid()) = owner_id);
create policy "owners read conversation messages" on public.conversation_messages for select to authenticated
using ((select auth.uid()) = owner_id);
create policy "owners read artifact versions" on public.artifact_versions for select to authenticated
using ((select auth.uid()) = owner_id);

-- The bucket is private. Only the Python API/worker uses the service-role key.
insert into storage.buckets (id, name, public)
values ('saral-documents', 'saral-documents', false)
on conflict (id) do update set public = false;

create policy "owners read their storage prefix" on storage.objects for select to authenticated
using (bucket_id = 'saral-documents' and (storage.foldername(name))[1] = (select auth.uid())::text);
