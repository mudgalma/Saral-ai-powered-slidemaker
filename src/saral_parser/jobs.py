"""Durable ingestion orchestration and Celery worker entry points."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import UUID, uuid4

from celery import Celery
from langsmith import trace, traceable

from .chunking import HybridDocumentChunker
from .docling_service import DoclingParser
from .embeddings import OpenAIEmbedder, vector_literal
from .exceptions import JobDispatchError
from .models import (
    AcceptedDocument,
    ChunkOptions,
    EmbeddingSettings,
    InputMetadata,
    ParseOptions,
    ParserSettings,
    SupabaseSettings,
)
from .persistence import SupabasePersistence
from .validation import save_upload, validate_local_pdf

LOGGER = logging.getLogger(__name__)


def create_celery(settings: SupabaseSettings | None = None) -> Celery:
    """Create a JSON-only Celery app backed by Redis."""
    redis_url = (
        settings.redis_url.get_secret_value()
        if settings
        else os.environ.get("SARAL_REDIS_URL", "redis://127.0.0.1:6379/0")
    )
    visibility_timeout = (
        settings.celery_visibility_timeout_seconds
        if settings
        else int(os.environ.get("SARAL_CELERY_VISIBILITY_TIMEOUT", "3600"))
    )
    app = Celery("saral_parser", broker=redis_url, backend=redis_url)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_soft_time_limit=visibility_timeout - 60,
        task_time_limit=visibility_timeout,
        broker_transport_options={"visibility_timeout": visibility_timeout},
        result_expires=3600,
    )
    return app


celery_app = create_celery()


class CeleryDispatcher:
    """Small API-side adapter so dispatch is replaceable in tests."""

    def dispatch(
        self,
        document_id: str,
        job_id: UUID,
        owner_id: UUID,
        parse_options: ParseOptions,
        chunk_options: ChunkOptions,
    ) -> None:
        try:
            process_document.apply_async(
                args=[
                    document_id,
                    str(job_id),
                    str(owner_id),
                    parse_options.model_dump(mode="json"),
                    chunk_options.model_dump(mode="json"),
                ],
                task_id=str(job_id),
            )
        except Exception as exc:
            raise JobDispatchError("Could not enqueue the document job") from exc

    def dispatch_embedding(self, document_id: str, job_id: UUID, owner_id: UUID) -> None:
        """Enqueue the independent retrieval-index job after parser persistence."""
        try:
            embed_document.apply_async(
                args=[document_id, str(job_id), str(owner_id)], task_id=str(job_id)
            )
        except Exception as exc:
            raise JobDispatchError("Could not enqueue the embedding job") from exc


class IngestionService:
    """Accept uploads durably before dispatching expensive parsing work."""

    def __init__(
        self,
        parser_settings: ParserSettings,
        persistence: SupabasePersistence,
        dispatcher: Any,
    ) -> None:
        self.parser_settings = parser_settings
        self.persistence = persistence
        self.dispatcher = dispatcher

    def accept(
        self,
        stream: Any,
        filename: str,
        owner_id: UUID,
        parse_options: ParseOptions,
        chunk_options: ChunkOptions,
    ) -> AcceptedDocument:
        with trace(
            "upload.validate_and_hash",
            run_type="tool",
            inputs={"filename": filename},
            tags=["saral", "api", "validation"],
        ):
            metadata = save_upload(stream, filename, self.parser_settings)
        metadata = metadata.model_copy(
            update={"document_id": f"doc_{owner_id.hex[:8]}_{metadata.content_sha256[:16]}"}
        )
        job_id = uuid4()
        trace_metadata = {"document_id": metadata.document_id, "job_id": str(job_id)}
        with trace(
            "supabase.persist_queued_document",
            run_type="tool",
            inputs={
                **trace_metadata,
                "size_bytes": metadata.size_bytes,
                "media_type": metadata.media_type,
                "parse_options": parse_options.model_dump(mode="json"),
            },
            metadata=trace_metadata,
            tags=["saral", "api", "supabase"],
        ):
            self.persistence.create_queued_document(
                owner_id=owner_id,
                job_id=job_id,
                metadata=metadata,
                options=parse_options,
            )
        # Supabase now owns the durable immutable copy; do not retain production uploads.
        Path(metadata.source_path).unlink(missing_ok=True)
        try:
            with trace(
                "celery.dispatch",
                run_type="tool",
                inputs={
                    **trace_metadata,
                    "chunk_options": chunk_options.model_dump(mode="json"),
                },
                metadata=trace_metadata,
                tags=["saral", "api", "queue"],
            ):
                self.dispatcher.dispatch(
                    metadata.document_id, job_id, owner_id, parse_options, chunk_options
                )
        except Exception:
            self.persistence.mark_failed(
                metadata.document_id, job_id, "dispatch_failed", "Could not enqueue parser job"
            )
            raise
        return AcceptedDocument(
            document_id=metadata.document_id,
            job_id=job_id,
            status_url=f"/v1/documents/{metadata.document_id}",
        )


@celery_app.task(bind=True, name="saral_parser.process_document")
@traceable(
    name="saral_parse_and_chunk",
    run_type="chain",
    process_inputs=lambda inputs: {
        "document_id": inputs.get("document_id"),
        "job_id": inputs.get("job_id"),
        "parse_options": inputs.get("parse_options"),
        "chunk_options": inputs.get("chunk_options"),
    },
    tags=["saral", "worker", "parse-and-chunk"],
)
def process_document(
    self: Any,
    document_id: str,
    job_id: str,
    owner_id: str,
    parse_options: dict[str, Any],
    chunk_options: dict[str, Any],
) -> dict[str, Any]:
    """Download, parse, chunk, validate, persist, and update status."""
    supabase_settings = SupabaseSettings.from_env()
    persistence = SupabasePersistence(supabase_settings)
    parsed_job_id = UUID(job_id)
    parsed_owner_id = UUID(owner_id)
    try:
        trace_metadata = {"document_id": document_id, "job_id": job_id}
        with trace(
            "worker.load_job",
            run_type="tool",
            inputs=trace_metadata,
            metadata=trace_metadata,
            tags=["saral", "worker", "supabase"],
        ):
            job = persistence.get_job(parsed_job_id)
        if not job or job.get("document_id") != document_id:
            raise JobDispatchError("Persisted job identity does not match task payload")
        with trace(
            "worker.mark_processing",
            run_type="tool",
            inputs=trace_metadata,
            metadata=trace_metadata,
            tags=["saral", "worker", "supabase"],
        ):
            persistence.mark_processing(document_id, parsed_job_id)
        with TemporaryDirectory() as temporary:
            worker_root = Path(temporary)
            parser_settings = ParserSettings.from_workspace(worker_root)
            parser_settings.upload_folder.mkdir(parents=True, exist_ok=True)
            local_path = parser_settings.upload_folder / "paper.pdf"
            storage_path = persistence.original_object_path(parsed_owner_id, document_id)
            with trace(
                "download_and_validate_original",
                run_type="tool",
                inputs={"document_id": document_id},
                metadata=trace_metadata,
                tags=["saral", "worker", "storage"],
            ):
                persistence.download_original(storage_path, local_path)
                metadata = validate_local_pdf(local_path, parser_settings)
            metadata = InputMetadata(
                **{
                    **metadata.model_dump(),
                    "document_id": document_id,
                    "original_filename": job.get("configuration", {}).get(
                        "original_filename", "paper.pdf"
                    ),
                }
            )
            with trace(
                "docling_parse_export_validate",
                run_type="tool",
                inputs={"document_id": document_id, "options": parse_options},
                metadata=trace_metadata,
                tags=["saral", "worker", "docling"],
            ):
                manifest = DoclingParser(parser_settings).parse_metadata(
                    metadata, ParseOptions.model_validate(parse_options)
                )
            job_dir = parser_settings.output_folder / document_id
            with trace(
                "docling_hybrid_chunk_and_validate",
                run_type="tool",
                inputs={"document_id": document_id, "options": chunk_options},
                metadata=trace_metadata,
                tags=["saral", "worker", "chunking"],
            ):
                chunk_result = HybridDocumentChunker(
                    ChunkOptions.model_validate(chunk_options)
                ).chunk_path(document_id, job_dir / "document.json")
            manifest.warnings.extend(chunk_result.warnings)
            with trace(
                "persist_parser_bundle",
                run_type="tool",
                inputs={
                    "document_id": document_id,
                    "asset_counts": manifest.counts,
                    "chunk_count": len(chunk_result.chunks),
                },
                metadata=trace_metadata,
                tags=["saral", "worker", "supabase"],
            ):
                persistence.persist_completed_bundle(
                    owner_id=parsed_owner_id,
                    job_id=parsed_job_id,
                    manifest=manifest,
                    job_dir=job_dir,
                    chunks=chunk_result.chunks,
                    chunk_configuration=chunk_result.options.model_dump(mode="json"),
                )
            embedding_status = _queue_embedding_if_configured(
                persistence=persistence,
                document_id=document_id,
                owner_id=parsed_owner_id,
            )
            return {
                "document_id": document_id,
                "status": "ready",
                "chunks": len(chunk_result.chunks),
                "embedding_status": embedding_status,
            }
    except Exception as exc:
        LOGGER.exception(
            "Document job failed", extra={"document_id": document_id, "job_id": job_id}
        )
        try:
            persistence.mark_failed(document_id, parsed_job_id, type(exc).__name__, str(exc))
        except Exception:
            LOGGER.exception("Could not persist failed job status")
        raise


def _queue_embedding_if_configured(
    *, persistence: SupabasePersistence, document_id: str, owner_id: UUID
) -> str:
    """Queue indexing without letting a provider configuration issue fail parsing."""
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        LOGGER.warning(
            "Embedding was not queued because OPENAI_API_KEY is absent",
            extra={"document_id": document_id, "owner_id": str(owner_id)},
        )
        return "not_started"
    try:
        embedding_settings = EmbeddingSettings.from_env()
        embedding_job_id = uuid4()
        persistence.create_embedding_job(
            document_id, owner_id, embedding_job_id, embedding_settings.model
        )
        CeleryDispatcher().dispatch_embedding(document_id, embedding_job_id, owner_id)
        return "queued"
    except Exception:
        # Parsing has already completed. Indexing status remains independently retryable.
        LOGGER.exception(
            "Could not queue embedding job",
            extra={"document_id": document_id, "owner_id": str(owner_id)},
        )
        try:
            persistence.mark_embedding_failed(
                document_id,
                owner_id,
                embedding_job_id if "embedding_job_id" in locals() else uuid4(),
                "Retrieval indexing could not be queued",
            )
        except Exception:
            LOGGER.exception(
                "Could not persist embedding dispatch failure",
                extra={"document_id": document_id, "owner_id": str(owner_id)},
            )
        return "failed"


@celery_app.task(bind=True, name="saral_parser.embed_document")
@traceable(
    name="saral_embed_document",
    run_type="chain",
    process_inputs=lambda inputs: {
        "document_id": inputs.get("document_id"),
        "job_id": inputs.get("job_id"),
    },
    tags=["saral", "worker", "embedding"],
)
def embed_document(
    self: Any, document_id: str, job_id: str, owner_id: str
) -> dict[str, Any]:
    """Embed one ready document's chunks and preserve its parser provenance."""
    supabase_settings = SupabaseSettings.from_env()
    persistence = SupabasePersistence(supabase_settings)
    parsed_job_id = UUID(job_id)
    parsed_owner_id = UUID(owner_id)
    try:
        job = persistence.get_job(parsed_job_id)
        if (
            not job
            or job.get("document_id") != document_id
            or job.get("owner_id") != str(parsed_owner_id)
            or job.get("job_type") != "embed_document"
        ):
            raise JobDispatchError("Persisted embedding job identity does not match task payload")
        embedding_settings = EmbeddingSettings.from_env()
        persistence.mark_embedding_processing(document_id, parsed_owner_id, parsed_job_id)
        chunks = persistence.get_chunks_for_embedding(
            document_id, parsed_owner_id, embedding_settings.model
        )
        if chunks:
            vectors = OpenAIEmbedder(embedding_settings).embed_texts(
                [str(chunk["contextualized_text"]) for chunk in chunks]
            )
            rows = [
                {"chunk_id": str(chunk["id"]), "embedding": vector_literal(vector)}
                for chunk, vector in zip(chunks, vectors)
            ]
            for start in range(0, len(rows), embedding_settings.batch_size):
                persistence.write_chunk_embeddings(
                    document_id,
                    parsed_owner_id,
                    embedding_settings.model,
                    rows[start : start + embedding_settings.batch_size],
                )
        persistence.mark_embedding_ready(document_id, parsed_owner_id, parsed_job_id)
        LOGGER.info(
            "Document embedding completed",
            extra={
                "document_id": document_id,
                "job_id": job_id,
                "embedded_count": len(chunks),
            },
        )
        return {"document_id": document_id, "status": "ready", "embedded_count": len(chunks)}
    except Exception:
        LOGGER.exception("Embedding job failed", extra={"document_id": document_id, "job_id": job_id})
        try:
            persistence.mark_embedding_failed(
                document_id,
                parsed_owner_id,
                parsed_job_id,
                "Retrieval indexing failed; retry the document index.",
            )
        except Exception:
            LOGGER.exception(
                "Could not persist embedding failure",
                extra={"document_id": document_id, "job_id": job_id},
            )
        raise
