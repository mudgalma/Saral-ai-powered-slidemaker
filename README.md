# SARAL document parser

Architecture diagrams: [overview](docs/architecture.md) and [full architecture + LLD](docs/full-architecture.md).

SARAL accepts one research-paper PDF at a time, validates it, converts it with Docling, and applies Docling's `HybridChunker`. The production path persists private files in Supabase Storage, queryable metadata and ordered chunks in Supabase Postgres, runs parsing and embedding separately with Celery/Redis, and performs hybrid pgvector + Postgres full-text retrieval with provenance.

The input PDF is data, never an instruction source. The parser does not modify a PDF placed in `paper_folder/`.

## Learning sequence followed

### 1. Convert into a Docling document

Followed: [Docling Quickstart - Python](https://docling-project.github.io/docling/getting_started/quickstart/#python).

The documented core workflow is `DocumentConverter().convert(source).document`, then an export. SARAL keeps that `DoclingDocument` as its primary, lossless JSON output instead of flattening it into plain text.

### 2. Select a research-paper input and output contract

Followed: [Supported formats](https://docling-project.github.io/docling/usage/supported_formats/).

Docling supports many inputs and lossless JSON, Markdown, HTML, and text outputs. SARAL currently exposes **PDF only**: that keeps upload validation, OCR, layout, table, figure, and formula behavior explicit for research papers. The module can be extended to the other documented formats only after their input-specific safety/validation policies are defined.

### 3. Preserve structure, reading order, and provenance

Followed: [Docling Document](https://docling-project.github.io/docling/concepts/docling_document/).

`document.json` is a lossless `DoclingDocument`. Its `body` tree and ordered children represent reading order and section hierarchy. `texts`, `tables`, and `pictures` are top-level typed item collections. Item records in `figures.json`, `tables.json`, and `formulas.json` retain Docling source references, page numbers, bounding boxes, and character spans where Docling supplies them.

### 4. Configure enrichments deliberately

Followed: [Formula and picture enrichment](https://docling-project.github.io/docling/usage/enrichments/) and the documented [custom conversion](https://docling-project.github.io/docling/_generated/examples/custom_convert/), [figure export](https://docling-project.github.io/docling/_generated/examples/export_figures/), [table export](https://docling-project.github.io/docling/_generated/examples/export_tables/), and [formula example](https://docling-project.github.io/docling/_generated/examples/code_formula_granite_docling/).

Docling documents that enrichment models are normally disabled because they add model executions and can materially increase runtime. SARAL exposes them rather than enabling everything by default:

| Feature | Setting | Purpose | Cost / boundary |
| --- | --- | --- | --- |
| OCR | `enable_ocr`, `ocr_engine`, `ocr_languages` | Read scanned or raster-only text | Extra OCR execution; use only for documents that need it. |
| Tables | `extract_table_structure` | Preserve cells as CSV and HTML plus table image | Enabled by default; structural predictions still require review. |
| Formula enrichment | `enable_formula_enrichment` | Convert detected formula items to LaTeX | Extra model work; enabled in the supplied-paper verification run. |
| Figure classification | `enable_picture_classification` | Classify document figures/charts/diagrams | Extra classifier execution; off by default. |
| Local figure description | `picture_description_mode=smolvlm_local` or `granite_local` | Generate a VLM description | Extra local model execution/download; generated text is never merged into source captions. |

Remote picture descriptions are intentionally not implemented. Docling documents that remote description sends document data to a provider; SARAL requires an explicit future privacy/security policy before adding that capability.

## Setup

Requires Python 3.9+. The dependency lock was produced from the tested environment.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

The exact direct dependencies are pinned in `pyproject.toml`; the resolved environment is in `requirements.lock`. On this macOS environment, set a writable bytecode cache when running commands:

```bash
export PYTHONPYCACHEPREFIX=/private/tmp/saral_pycache
```

## Parse a paper locally

Put a PDF in `paper_folder/` and run:

```bash
.venv/bin/saral-parse paper_folder/my-paper.pdf --workspace . \
  --options-json '{"enable_formula_enrichment": true}'
```

A scanned English paper can use OCR:

```bash
.venv/bin/saral-parse paper_folder/scanned-paper.pdf --workspace . \
  --options-json '{"enable_ocr": true, "ocr_engine": "ocrmac", "ocr_languages": ["en"]}'
```

`ocrmac` is macOS-specific. `auto` uses Docling's configured default OCR implementation; `tesseract_cli` requires a working local Tesseract executable. OCR is not an assertion of text accuracy—review page images and the validation report.

## Production services

Create a Supabase project, review and apply [supabase/schema.sql](supabase/schema.sql) followed by
[supabase/retrieval.sql](supabase/retrieval.sql), then set
server-only values (never use the service-role key in `Frontend/`):

```bash
export SUPABASE_URL=https://PROJECT.supabase.co
export SUPABASE_SERVICE_ROLE_KEY=server-only-secret
export SARAL_REDIS_URL=redis://127.0.0.1:6379/0
export OPENROUTER_API_KEY=server-only-secret
# Optional embedding default
export SARAL_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
export SARAL_EMBEDDING_MODEL=openai/text-embedding-3-small
export SARAL_GENERATION_MODEL=openai/gpt-4.1-mini
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=server-only-secret
export LANGSMITH_PROJECT=saral-parser
```

Start Redis, Celery, and FastAPI:

```bash
docker compose up -d redis
.venv/bin/celery -A saral_parser.jobs:celery_app worker --loglevel=INFO
.venv/bin/uvicorn saral_parser.app:create_app --factory --host 127.0.0.1 --port 8000
```

The API verifies a Supabase user JWT. For local development only, you may set
`SARAL_ENVIRONMENT=development` and `SARAL_DEV_USER_ID=<an auth.users UUID>` instead.
The current frontend upload adapter reads the short-lived user token from session storage key
`saral_supabase_access_token`; wire the eventual Supabase Auth screen to that session boundary.
LangSmith tracing is opt-in through its environment variables. Traces receive IDs, options,
counts, durations, and failures—not PDF text or credentials. Local structured `debug.jsonl`
continues to provide detailed parser-stage evidence.

### LangSmith demo view

Set the LangSmith variables in **both** the FastAPI and Celery terminals, submit a fresh PDF,
then open the `saral-parser` project in LangSmith. Filter by the returned `document_id` or
`job_id`. You will see two correlated traces because Redis is an asynchronous process boundary:

- `api.accept_document`: upload metadata, validation/hash, queued Supabase record, and dispatch.
- `saral_parse_and_chunk`: job loading/status, private download, Docling conversion, export,
  parser validation, hybrid chunking/invariant counts, individual artifact uploads, batch row
  counts, final persistence, and failures.
- `saral_embed_document`: independent chunk-vector indexing. It logs only document/job IDs,
  batch counts, timings, and errors—not chunk text or vectors.

Useful tags are `api`, `worker`, `docling`, `chunking`, `validation`, `storage`, `database`, and
`supabase`. Inputs are deliberately metadata-only: filenames, identifiers, byte/count metrics,
and feature options. JWTs, service-role keys, PDF bytes, full extracted text, and chunk text are
not traced. This keeps the demo operationally detailed without copying client papers into the
observability provider.

## FastAPI contract

Start the thin transport layer:

```bash
.venv/bin/uvicorn saral_parser.app:create_app --factory --reload
```

Submit a PDF and optional JSON options:

```bash
curl -X POST http://127.0.0.1:8000/v1/documents \
  -H "Authorization: Bearer $SUPABASE_USER_ACCESS_TOKEN" \
  -F 'file=@paper_folder/my-paper.pdf;type=application/pdf' \
  -F 'options={"enable_formula_enrichment":true}' \
  -F 'chunk_options={"max_tokens":512,"merge_peers":true}'
```

The upload returns `202 Accepted` with `document_id`, `job_id`, and `status_url`. Poll status with:

```bash
curl -H "Authorization: Bearer $SUPABASE_USER_ACCESS_TOKEN" \
  http://127.0.0.1:8000/v1/documents/doc_0123456789abcdef
```

When `embedding_status` is `ready`, retrieve grounded evidence:

```bash
curl -X POST http://127.0.0.1:8000/v1/documents/doc_0123456789abcdef/retrieve \
  -H "Authorization: Bearer $SUPABASE_USER_ACCESS_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question":"What methods does this paper use?","top_k":6}'
```

The response returns the top fused chunks with chunk IDs, text, headings, page numbers, source
references, full provenance, figure/table asset IDs, and dense/sparse ranks. This is the grounded
context contract for the next generation milestone.

Generate a citation-grounded artifact from that existing hybrid retrieval index:

```bash
curl -X POST http://127.0.0.1:8000/v1/documents/doc_0123456789abcdef/generate \
  -H "Authorization: Bearer $SUPABASE_USER_ACCESS_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"artifact_type":"script","audience":"Policymakers","length":"standard", "style":"Plain English", "user_instruction":"Create a 90-second script about the method."}'
```

`POST /generate` retrieves up to six owner-scoped chunks, creates an `EvidencePack`, asks the
configured OpenRouter model for JSON-schema constrained output, then validates word budget, visible
chunk-ID citations, and claim-to-evidence mappings. It makes at most one corrective regeneration;
otherwise it returns a safe `flagged` response without an artifact. See
[`docs/generation-workflow.md`](docs/generation-workflow.md) for the workflow and documentation
traceability.

For a persistent conversational turn (new artifact, revision, or document question), send the
browser thread UUID and optional UI choices. The service records conversation state and immutable
artifact versions server-side, resolves `#N` and “this slide” revision references, then retrieves
document evidence again before creating a version delta:

```bash
curl -X POST http://127.0.0.1:8000/v1/documents/doc_0123456789abcdef/conversations/messages \
  -H "Authorization: Bearer $SUPABASE_USER_ACCESS_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"thread_id":"22222222-2222-4222-8222-222222222222", "message":"Make #1 shorter", "audience":"Policymakers", "length":"brief", "style":"Plain English"}'
```

## Connected frontend upload

The React frontend in `Frontend/` is connected to the parser endpoint. It accepts one PDF, sends the original browser file as multipart form data, and renders the returned document ID, filename, page/figure/table/formula counts, and warnings. The retrieval endpoint is ready for the workspace UI to consume.

Run the two services in separate terminals:

```bash
# Terminal 1: parser API (also run Redis and the Celery worker shown above)
SARAL_ALLOWED_ORIGINS=http://localhost:8080,http://127.0.0.1:8080 \
  .venv/bin/uvicorn saral_parser.app:create_app --factory --host 127.0.0.1 --port 8000

# Terminal 2: React UI
cd Frontend
cp .env.example .env.local
npm install
npm run dev
```

`VITE_SARAL_PARSER_API_URL` is a public browser configuration, not a secret. For deployed environments, set it to the HTTPS parser API URL and set `SARAL_ALLOWED_ORIGINS` to the exact HTTPS frontend origin(s), comma-separated. Do not use `*` for parser uploads.

### Browser input and result

The user selects exactly one PDF up to 50 MiB and optionally writes a message. The frontend sends:

```text
POST /v1/documents
Content-Type: multipart/form-data

file: <the selected PDF bytes>
options: { enable_formula_enrichment: true, extract_table_structure: true,
           generate_page_images: true, generate_picture_images: true }
```

FastAPI responds immediately with a queued job. The frontend polls until `ready` or `failed`, then renders the filename, document ID, parser counts, chunk count, warnings, and errors.

## Hybrid chunking contract

`document.json` is loaded back into a `DoclingDocument`. `HybridChunker` first follows its
hierarchical structure and then splits/merges against the configured tokenizer budget. Each
Postgres `document_chunks` row has deterministic `chunk_index`/`id`, raw and contextualized
text, headings, captions, source refs, page/bounding-box provenance, related asset IDs, content
types, token count, and chunker version. Validation fails the job if source order is reversed,
a formula source is missing/duplicated, grounding refs are absent, or a non-atomic chunk exceeds
the limit. Repeated table refs are allowed because a large atomic table may span adjacent chunks.

The saved actual-paper preview is
[`outputs/doc_bdfaa68d8984f0dc/chunks.preview.json`](outputs/doc_bdfaa68d8984f0dc/chunks.preview.json):
42 chunks, produced by a real offline `HybridChunker` run over the existing Docling JSON.

## Output contract

For local CLI inspection, each job is written to `outputs/<document_id>/`. Production workers
use an isolated temporary directory and remove it after successful/failed processing.

Production placement:

- Storage: `{owner_id}/{document_id}/original/paper.pdf`, `parsed/document.json`, `assets/**`,
  and private `diagnostics/{manifest,validation_report,debug}` files.
- Postgres: `documents`, `processing_jobs`, `document_assets`, `document_validations`, and
  ordered `document_chunks` rows.
- Redis: transient Celery delivery/result state only—not the authoritative document record.

| Artifact | Contract |
| --- | --- |
| `manifest.json` | Job identity, content hash, exact options, status, counts, warnings, and artifact paths. |
| `document.json` | Primary Docling lossless structured output for later downstream integration. |
| `document.md`, `document.html`, `document.txt` | Readable exports. Markdown/HTML embed images so a moved bundle does not have dangling converter-internal references. |
| `figures.json` | One record per `PictureItem`: Docling source ref, provenance, original caption only when explicitly linked by Docling, and separate generated description. |
| `tables.json`, `assets/tables/` | Table source refs/provenance and CSV, HTML, plus rendered image when available. |
| `formulas.json` | Formula source refs/provenance and Docling's LaTeX text. |
| `assets/figures`, `assets/pages` | Rendered image assets when Docling makes them available. |
| `validation_report.json` | Artifact failures, warnings, and review items. It does not fabricate semantic correctness. |
| `debug.jsonl` | Structured stage logs: start/completion/failure, durations, configuration, counts, warnings, and locations. |
| `intermediate/` | Input metadata, exact pipeline options, bounded structural preview, and an opt-in full diagnostic export. |

Document IDs are `doc_<first 16 hex chars of SHA-256>` and represent content identity. Reprocessing identical bytes intentionally uses the same ID and refreshes its bundle; the JSONL log preserves attempts.

## Safety and limits

- Only regular `.pdf` files below 50 MiB are accepted, with both extension and `%PDF-` signature checks.
- Local parser inputs must resolve inside `paper_folder/` or `uploads/`; symlinks and traversal are rejected.
- Uploads are streamed with a byte limit and stored only under `uploads/` using a sanitized name.
- Artifact retrieval is constrained to its document's output directory.
- No secrets are logged. Document diagnostics are bounded by default. Set `debug_full_document_logging: true` only for controlled debugging; it writes an opt-in full Markdown diagnostic under the job's `intermediate/` folder.

## Verification run: `1706.03762v7.pdf`

Source: `paper_folder/1706.03762v7.pdf`, SHA-256 `bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697`.

Actual (not mocked) Docling run:

- Configuration: table structure and page/picture assets enabled; formula enrichment enabled; OCR, picture classification, and generated descriptions disabled.
- Result: 15 pages, 6 pictures, 4 tables, and 5 formulas in 43.216 seconds.
- Validation: no failures. Warnings accurately identify disabled OCR and picture descriptions.
- Visual comparison: the extracted Figure 2 image matches its page-4 source region and its original caption is preserved in `figure_0003`; table 1's rendered asset and 4-column CSV match the page-6 table; the page-4 attention equation is represented by `formula_0001` with source element `#/texts/46` and page/bounding-box provenance. Formula LaTex is model output and should be reviewed for typographic normalization (for example, spacing between symbols) before any scientific reuse.

The environment did not contain Poppler, so source-page comparison used the page images rendered and retained by Docling during the same conversion rather than an independent PDF renderer. This limitation is recorded here rather than hidden.

## Test and inspect

```bash
PYTHONPYCACHEPREFIX=/private/tmp/saral_pycache .venv/bin/ruff check src tests
PYTHONPYCACHEPREFIX=/private/tmp/saral_pycache .venv/bin/pytest -q
```

The test suite is deliberately labeled: it uses fake services/documents for HTTP, validation, serialization, and provenance checks; the verification run above is the actual Docling/model run.

## Known limitations

- PDF support is intentionally the current API boundary, despite Docling's broader format support.
- Detection and enrichment results are not ground truth. Use `validation_report.json`, page images, bounding boxes, and source refs for review.
- Captions are recorded only when Docling explicitly associates them with a picture. SARAL does not guess missing associations.
- No OCR run was performed for the supplied born-digital paper. A scanned paper needs a separate OCR-enabled run and review.
- No classification or VLM description run was performed; those options are configurable and their generated descriptions stay separate from authored captions.
- Document model downloads and local VLM/OCR availability depend on the installed Docling extras and host environment.
