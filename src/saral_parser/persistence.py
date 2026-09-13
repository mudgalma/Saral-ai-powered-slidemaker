"""Supabase persistence for immutable parser files and queryable metadata."""

from __future__ import annotations

import json
import mimetypes
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, TypeVar
from uuid import UUID

from langsmith import traceable

from .exceptions import PersistenceError
from .models import ArtifactManifest, DocumentChunk, InputMetadata, ParseOptions, SupabaseSettings

T = TypeVar("T")
_TRANSIENT_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


class SupabasePersistence:
    """Store parser files privately and normalized operational records in Postgres."""

    def __init__(self, settings: SupabaseSettings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client or self._create_client(settings)

    @staticmethod
    def _create_client(settings: SupabaseSettings) -> Any:
        try:
            from supabase import ClientOptions, create_client
        except ImportError as exc:  # pragma: no cover - installation boundary
            raise PersistenceError("The pinned supabase package is not installed") from exc
        options = ClientOptions(
            postgrest_client_timeout=settings.request_timeout_seconds,
            storage_client_timeout=settings.request_timeout_seconds,
            persist_session=False,
            auto_refresh_token=False,
        )
        return create_client(
            settings.url,
            settings.service_role_key.get_secret_value(),
            options=options,
        )

    @traceable(
        name="supabase.create_queued_document",
        run_type="tool",
        process_inputs=lambda inputs: {
            "document_id": getattr(inputs.get("metadata"), "document_id", None),
            "job_id": str(inputs.get("job_id")),
            "size_bytes": getattr(inputs.get("metadata"), "size_bytes", None),
        },
        tags=["saral", "supabase", "ingestion"],
    )
    def create_queued_document(
        self,
        *,
        owner_id: UUID,
        job_id: UUID,
        metadata: InputMetadata,
        options: ParseOptions,
    ) -> str:
        """Upload the immutable source and atomically upsert initial database records."""
        object_path = self.original_object_path(owner_id, metadata.document_id)
        self._upload_file(Path(metadata.source_path), object_path, "application/pdf")
        now = _utc_now()
        document = {
            "id": metadata.document_id,
            "owner_id": str(owner_id),
            "content_sha256": metadata.content_sha256,
            "original_filename": metadata.original_filename,
            "media_type": metadata.media_type,
            "size_bytes": metadata.size_bytes,
            "status": "queued",
            "original_storage_path": object_path,
            "parser_options": options.model_dump(mode="json"),
            "updated_at": now,
        }
        job = {
            "id": str(job_id),
            "document_id": metadata.document_id,
            "owner_id": str(owner_id),
            "job_type": "parse_and_chunk",
            "status": "queued",
            "configuration": {
                "parse_options": options.model_dump(mode="json"),
                "original_filename": metadata.original_filename,
            },
        }
        self._execute(lambda: self._client.table("documents").upsert(document).execute())
        self._execute(lambda: self._client.table("processing_jobs").insert(job).execute())
        return object_path

    def get_document(self, document_id: str, owner_id: UUID) -> Dict[str, Any] | None:
        """Return one owner-scoped document status record."""
        response = self._execute(
            lambda: self._client.table("documents")
            .select("*")
            .eq("id", document_id)
            .eq("owner_id", str(owner_id))
            .limit(1)
            .execute()
        )
        rows = getattr(response, "data", None) or []
        return dict(rows[0]) if rows else None

    def create_embedding_job(
        self, document_id: str, owner_id: UUID, job_id: UUID, embedding_model: str
    ) -> None:
        """Record an embedding job after validated chunks are durable."""
        job = {
            "id": str(job_id),
            "document_id": document_id,
            "owner_id": str(owner_id),
            "job_type": "embed_document",
            "status": "queued",
            "configuration": {"embedding_model": embedding_model},
        }
        now = _utc_now()
        self._execute(lambda: self._client.table("processing_jobs").insert(job).execute())
        self._execute(
            lambda: self._client.table("documents")
            .update(
                {
                    "embedding_status": "queued",
                    "embedding_model": embedding_model,
                    "embedding_error": None,
                    "updated_at": now,
                }
            )
            .eq("id", document_id)
            .eq("owner_id", str(owner_id))
            .execute()
        )

    def get_chunks_for_embedding(
        self, document_id: str, owner_id: UUID, embedding_model: str
    ) -> list[Dict[str, Any]]:
        """Return chunks that need vectors for the requested model revision."""
        response = self._execute(
            lambda: self._client.table("document_chunks")
            .select("id,contextualized_text,embedding,embedding_model")
            .eq("document_id", document_id)
            .eq("owner_id", str(owner_id))
            .order("chunk_index")
            .execute()
        )
        rows = getattr(response, "data", None) or []
        return [
            dict(row)
            for row in rows
            if not row.get("embedding") or row.get("embedding_model") != embedding_model
        ]

    def mark_embedding_processing(self, document_id: str, owner_id: UUID, job_id: UUID) -> None:
        """Mark one retrieval-index job in progress without changing parser readiness."""
        now = _utc_now()
        self._execute(
            lambda: self._client.table("processing_jobs")
            .update({"status": "processing", "started_at": now})
            .eq("id", str(job_id))
            .eq("document_id", document_id)
            .eq("owner_id", str(owner_id))
            .execute()
        )
        self._execute(
            lambda: self._client.table("documents")
            .update({"embedding_status": "processing", "updated_at": now})
            .eq("id", document_id)
            .eq("owner_id", str(owner_id))
            .execute()
        )

    def write_chunk_embeddings(
        self,
        document_id: str,
        owner_id: UUID,
        embedding_model: str,
        rows: Sequence[Dict[str, str]],
    ) -> None:
        """Write one validated batch of vectors through a server-only SQL function."""
        if not rows:
            return
        self._execute(
            lambda: self._client.rpc(
                "write_document_chunk_embeddings",
                {
                    "p_owner_id": str(owner_id),
                    "p_document_id": document_id,
                    "p_embedding_model": embedding_model,
                    "p_embeddings": list(rows),
                },
            ).execute()
        )

    def mark_embedding_ready(self, document_id: str, owner_id: UUID, job_id: UUID) -> None:
        """Mark a fully written vector index ready for retrieval."""
        now = _utc_now()
        self._execute(
            lambda: self._client.table("processing_jobs")
            .update({"status": "ready", "completed_at": now})
            .eq("id", str(job_id))
            .eq("document_id", document_id)
            .eq("owner_id", str(owner_id))
            .execute()
        )
        self._execute(
            lambda: self._client.table("documents")
            .update({"embedding_status": "ready", "embedding_error": None, "updated_at": now})
            .eq("id", document_id)
            .eq("owner_id", str(owner_id))
            .execute()
        )

    def mark_embedding_failed(
        self, document_id: str, owner_id: UUID, job_id: UUID, message: str
    ) -> None:
        """Persist a bounded indexing failure while leaving parsed chunks available."""
        now = _utc_now()
        safe_message = message[:1_000]
        self._execute(
            lambda: self._client.table("processing_jobs")
            .update(
                {
                    "status": "failed",
                    "error_code": "embedding_failed",
                    "error_message": safe_message,
                    "completed_at": now,
                }
            )
            .eq("id", str(job_id))
            .eq("document_id", document_id)
            .eq("owner_id", str(owner_id))
            .execute()
        )
        self._execute(
            lambda: self._client.table("documents")
            .update(
                {"embedding_status": "failed", "embedding_error": safe_message, "updated_at": now}
            )
            .eq("id", document_id)
            .eq("owner_id", str(owner_id))
            .execute()
        )

    def search_document_chunks_dense(
        self, document_id: str, owner_id: UUID, embedding: str, limit: int
    ) -> list[Dict[str, Any]]:
        """Run owner- and document-scoped cosine search through pgvector RPC."""
        response = self._execute(
            lambda: self._client.rpc(
                "search_document_chunks_dense",
                {
                    "p_owner_id": str(owner_id),
                    "p_document_id": document_id,
                    "p_query_embedding": embedding,
                    "p_match_count": limit,
                },
            ).execute()
        )
        return [dict(row) for row in (getattr(response, "data", None) or [])]

    def search_document_chunks_sparse(
        self, document_id: str, owner_id: UUID, question: str, limit: int
    ) -> list[Dict[str, Any]]:
        """Run owner- and document-scoped native full-text search through RPC."""
        response = self._execute(
            lambda: self._client.rpc(
                "search_document_chunks_sparse",
                {
                    "p_owner_id": str(owner_id),
                    "p_document_id": document_id,
                    "p_query_text": question,
                    "p_match_count": limit,
                },
            ).execute()
        )
        return [dict(row) for row in (getattr(response, "data", None) or [])]

    @traceable(
        name="supabase.get_job",
        run_type="tool",
        process_inputs=lambda inputs: {"job_id": str(inputs.get("job_id"))},
        process_outputs=lambda output: {
            "found": output is not None,
            "status": output.get("status") if output else None,
        },
        tags=["saral", "supabase", "job"],
    )
    def get_job(self, job_id: UUID) -> Dict[str, Any] | None:
        """Return a job for trusted worker-side processing."""
        response = self._execute(
            lambda: self._client.table("processing_jobs")
            .select("*")
            .eq("id", str(job_id))
            .limit(1)
            .execute()
        )
        rows = getattr(response, "data", None) or []
        return dict(rows[0]) if rows else None

    @traceable(
        name="supabase.mark_processing",
        run_type="tool",
        process_inputs=lambda inputs: {
            "document_id": inputs.get("document_id"),
            "job_id": str(inputs.get("job_id")),
        },
        tags=["saral", "supabase", "job"],
    )
    def mark_processing(self, document_id: str, job_id: UUID) -> None:
        """Mark a dispatched job and its document as processing."""
        now = _utc_now()
        self._execute(
            lambda: self._client.table("processing_jobs")
            .update({"status": "processing", "started_at": now})
            .eq("id", str(job_id))
            .execute()
        )
        self._execute(
            lambda: self._client.table("documents")
            .update({"status": "processing", "updated_at": now})
            .eq("id", document_id)
            .execute()
        )

    @traceable(
        name="supabase.download_original",
        run_type="tool",
        process_inputs=lambda inputs: {"storage_path": inputs.get("storage_path")},
        tags=["saral", "supabase", "storage"],
    )
    def download_original(self, storage_path: str, destination: Path) -> None:
        """Download one trusted object path into an isolated worker location."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self._execute(
            lambda: self._client.storage.from_(self.settings.bucket).download(storage_path)
        )
        if not isinstance(payload, bytes):
            raise PersistenceError("Supabase returned a non-bytes original document")
        destination.write_bytes(payload)

    @traceable(
        name="supabase.persist_completed_bundle",
        run_type="tool",
        process_inputs=lambda inputs: {
            "document_id": getattr(inputs.get("manifest"), "document_id", None),
            "job_id": str(inputs.get("job_id")),
            "chunk_count": len(inputs.get("chunks") or []),
        },
        tags=["saral", "supabase", "persistence"],
    )
    def persist_completed_bundle(
        self,
        *,
        owner_id: UUID,
        job_id: UUID,
        manifest: ArtifactManifest,
        job_dir: Path,
        chunks: Sequence[DocumentChunk],
        chunk_configuration: Mapping[str, Any],
    ) -> None:
        """Upload authoritative outputs, batch metadata, then mark the job ready."""
        prefix = self.document_prefix(owner_id, manifest.document_id)
        docling_path = f"{prefix}/parsed/document.json"
        self._upload_file(job_dir / "document.json", docling_path, "application/json")
        for filename, media_type in (
            ("manifest.json", "application/json"),
            ("validation_report.json", "application/json"),
            ("debug.jsonl", "application/x-ndjson"),
        ):
            self._upload_file(job_dir / filename, f"{prefix}/diagnostics/{filename}", media_type)

        assets = list(self._asset_rows(manifest.document_id, job_dir, prefix))
        for asset in assets:
            local_path = job_dir / asset.pop("local_relative_path")
            self._upload_file(local_path, asset["storage_path"], asset["media_type"])

        validation = json.loads((job_dir / "validation_report.json").read_text(encoding="utf-8"))
        if assets:
            self._batch_upsert("document_assets", assets)
        if chunks:
            self._batch_upsert(
                "document_chunks",
                [self._chunk_row(owner_id, job_id, chunk) for chunk in chunks],
            )

        now = _utc_now()
        self._execute(
            lambda: self._client.table("document_validations")
            .upsert(
                {
                    "document_id": manifest.document_id,
                    "job_id": str(job_id),
                    "owner_id": str(owner_id),
                    "status": validation["status"],
                    "warnings": validation.get("warnings", []),
                    "failures": validation.get("failures", []),
                    "review_items": validation.get("review_items", []),
                },
                on_conflict="document_id,job_id",
            )
            .execute()
        )
        self._execute(
            lambda: self._client.table("documents")
            .update(
                {
                    "status": "ready",
                    "docling_json_storage_path": docling_path,
                    "page_count": manifest.counts.get("pages", 0),
                    "figure_count": manifest.counts.get("pictures", 0),
                    "table_count": manifest.counts.get("tables", 0),
                    "formula_count": manifest.counts.get("formulas", 0),
                    "chunk_count": len(chunks),
                    "warnings": manifest.warnings,
                    "errors": [],
                    "chunker_options": dict(chunk_configuration),
                    "updated_at": now,
                }
            )
            .eq("id", manifest.document_id)
            .execute()
        )
        self._execute(
            lambda: self._client.table("processing_jobs")
            .update({"status": "ready", "completed_at": now})
            .eq("id", str(job_id))
            .execute()
        )

    @traceable(
        name="supabase.mark_failed",
        run_type="tool",
        process_inputs=lambda inputs: {
            "document_id": inputs.get("document_id"),
            "job_id": str(inputs.get("job_id")),
            "error_code": inputs.get("code"),
        },
        tags=["saral", "supabase", "failure"],
    )
    def mark_failed(self, document_id: str, job_id: UUID, code: str, message: str) -> None:
        """Persist a bounded client-safe failure without storing document content."""
        now = _utc_now()
        safe_message = message[:1000]
        self._execute(
            lambda: self._client.table("processing_jobs")
            .update(
                {
                    "status": "failed",
                    "error_code": code,
                    "error_message": safe_message,
                    "completed_at": now,
                }
            )
            .eq("id", str(job_id))
            .execute()
        )
        self._execute(
            lambda: self._client.table("documents")
            .update({"status": "failed", "errors": [safe_message], "updated_at": now})
            .eq("id", document_id)
            .execute()
        )

    def download_asset(self, document_id: str, asset_id: str, owner_id: UUID) -> tuple[bytes, str]:
        """Return an owner-scoped private asset for FastAPI proxy delivery."""
        response = self._execute(
            lambda: self._client.table("document_assets")
            .select("storage_path,media_type")
            .eq("id", asset_id)
            .eq("document_id", document_id)
            .eq("owner_id", str(owner_id))
            .limit(1)
            .execute()
        )
        rows = getattr(response, "data", None) or []
        if not rows:
            raise PersistenceError("Asset does not exist")
        row = rows[0]
        payload = self._execute(
            lambda: self._client.storage.from_(self.settings.bucket).download(row["storage_path"])
        )
        if not isinstance(payload, bytes):
            raise PersistenceError("Supabase returned a non-bytes asset")
        return payload, row["media_type"]

    def original_object_path(self, owner_id: UUID, document_id: str) -> str:
        return f"{self.document_prefix(owner_id, document_id)}/original/paper.pdf"

    @staticmethod
    def document_prefix(owner_id: UUID, document_id: str) -> str:
        return f"{owner_id}/{document_id}"

    @traceable(
        name="supabase.upload_artifact",
        run_type="tool",
        process_inputs=lambda inputs: {
            "filename": getattr(inputs.get("local_path"), "name", None),
            "object_path": inputs.get("object_path"),
            "media_type": inputs.get("media_type"),
        },
        tags=["saral", "supabase", "storage"],
    )
    def _upload_file(self, local_path: Path, object_path: str, media_type: str) -> None:
        if not local_path.is_file():
            raise PersistenceError(f"Required artifact is missing: {local_path.name}")
        self._execute(
            lambda: self._client.storage.from_(self.settings.bucket).upload(
                object_path,
                local_path,
                {"content-type": media_type, "upsert": "true"},
            )
        )

    def _asset_rows(self, document_id: str, job_dir: Path, prefix: str) -> Iterable[Dict[str, Any]]:
        metadata = self._asset_metadata(job_dir)
        for local_path in sorted((job_dir / "assets").rglob("*")):
            if not local_path.is_file():
                continue
            relative = local_path.relative_to(job_dir).as_posix()
            kind = local_path.parent.name.rstrip("s")
            stem = local_path.stem
            key = (kind, stem)
            item = metadata.get(key, {})
            yield {
                "id": f"{document_id}:{kind}:{local_path.name}",
                "document_id": document_id,
                "owner_id": prefix.split("/", 1)[0],
                "asset_type": kind,
                "storage_path": f"{prefix}/{relative}",
                "media_type": mimetypes.guess_type(local_path.name)[0]
                or "application/octet-stream",
                "page_number": item.get("page_number") or _page_from_name(stem),
                "source_ref": item.get("source_element"),
                "bounding_box": item.get("bbox"),
                "caption": item.get("caption", []),
                "generated_description": item.get("generated_description"),
                "local_relative_path": relative,
            }

    @staticmethod
    def _asset_metadata(job_dir: Path) -> Dict[tuple[str, str], Dict[str, Any]]:
        result: Dict[tuple[str, str], Dict[str, Any]] = {}
        for kind, filename in (("figure", "figures.json"), ("table", "tables.json")):
            path = job_dir / filename
            if not path.is_file():
                continue
            for item in json.loads(path.read_text(encoding="utf-8")):
                provenance = item.get("provenance") or [{}]
                first = provenance[0]
                result[(kind, item["id"])] = {
                    "source_element": item.get("source_element"),
                    "page_number": first.get("page_number"),
                    "bbox": first.get("bbox"),
                    "caption": item.get("original_caption", []),
                    "generated_description": item.get("generated_description"),
                }
        return result

    @staticmethod
    def _chunk_row(owner_id: UUID, job_id: UUID, chunk: DocumentChunk) -> Dict[str, Any]:
        return {
            "id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "owner_id": str(owner_id),
            "job_id": str(job_id),
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "contextualized_text": chunk.contextualized_text,
            "headings": chunk.headings,
            "captions": chunk.captions,
            "source_refs": chunk.source_refs,
            "page_numbers": chunk.page_numbers,
            "provenance": chunk.provenance,
            "asset_ids": chunk.asset_ids,
            "content_types": chunk.content_types,
            "token_count": chunk.token_count,
            "chunker_version": chunk.chunker_version,
        }

    @traceable(
        name="supabase.batch_upsert",
        run_type="tool",
        process_inputs=lambda inputs: {
            "table": inputs.get("table"),
            "row_count": len(inputs.get("rows") or []),
        },
        tags=["saral", "supabase", "database"],
    )
    def _batch_upsert(self, table: str, rows: Sequence[Dict[str, Any]]) -> None:
        for start in range(0, len(rows), 500):
            batch = list(rows[start : start + 500])
            self._execute(
                lambda batch=batch: self._client.table(table)
                .upsert(batch, on_conflict="id")
                .execute()
            )

    def _execute(self, operation: Callable[[], T]) -> T:
        for attempt in range(3):
            try:
                return operation()
            except Exception as exc:
                if attempt == 2 or not _is_transient(exc):
                    raise PersistenceError("Supabase persistence operation failed") from exc
                time.sleep(0.25 * (2**attempt))
        raise AssertionError("unreachable")


def _is_transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    try:
        return int(status) in _TRANSIENT_STATUS_CODES
    except (TypeError, ValueError):
        return isinstance(exc, (ConnectionError, TimeoutError))


def _page_from_name(stem: str) -> int | None:
    if not stem.startswith("page_"):
        return None
    try:
        return int(stem.removeprefix("page_"))
    except ValueError:
        return None


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
