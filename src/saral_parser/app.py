"""Thin FastAPI transport for asynchronous document ingestion."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import UUID

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from langsmith import trace
from pydantic import ValidationError

from .embeddings import OpenAIEmbedder
from .exceptions import ConfigurationError, InputValidationError, PersistenceError, RetrievalError
from .jobs import CeleryDispatcher, IngestionService
from .models import (
    AcceptedDocument,
    ChunkOptions,
    DocumentStatus,
    EmbeddingSettings,
    ParseOptions,
    ParserSettings,
    RetrievalRequest,
    RetrievalResponse,
    SupabaseSettings,
)
from .persistence import SupabasePersistence
from .retrieval import HybridRetriever
from .validation import ensure_directories, validate_document_id

UserResolver = Callable[[Optional[str]], UUID]


def create_app(
    settings: ParserSettings | None = None,
    persistence: Any | None = None,
    dispatcher: Any | None = None,
    user_resolver: UserResolver | None = None,
    retriever: Any | None = None,
) -> FastAPI:
    """Build the API with injectable boundaries for deterministic tests."""
    active_settings = settings or ParserSettings.from_workspace(Path.cwd())
    ensure_directories(active_settings)
    active_persistence = persistence or SupabasePersistence(SupabaseSettings.from_env())
    ingestion = IngestionService(
        active_settings, active_persistence, dispatcher or CeleryDispatcher()
    )
    resolve_user = user_resolver or _supabase_user_resolver(active_persistence)

    app = FastAPI(title="SARAL document ingestion API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(active_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    def current_user(authorization: Optional[str] = Header(default=None)) -> UUID:
        try:
            return resolve_user(authorization)
        except Exception as exc:
            raise HTTPException(
                status_code=401, detail="A valid Supabase access token is required"
            ) from exc

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "service": "saral-document-parser"}

    @app.post(
        "/v1/documents", response_model=AcceptedDocument, status_code=status.HTTP_202_ACCEPTED
    )
    def create_document(
        file: UploadFile = File(...),
        options: str = Form(default="{}"),
        chunk_options: str = Form(default="{}"),
        owner_id: UUID = Depends(current_user),
    ) -> AcceptedDocument:
        try:
            parse_config = ParseOptions.model_validate(json.loads(options))
            chunk_config = ChunkOptions.model_validate(json.loads(chunk_options))
            if not file.filename:
                raise InputValidationError("Uploaded file must have a filename")
            with trace(
                "api.accept_document",
                run_type="chain",
                inputs={
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "parse_options": parse_config.model_dump(mode="json"),
                    "chunk_options": chunk_config.model_dump(mode="json"),
                },
                tags=["saral", "api", "upload"],
            ):
                return ingestion.accept(
                    file.file, file.filename, owner_id, parse_config, chunk_config
                )
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Options must be JSON objects") from exc
        except (InputValidationError, ValidationError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/documents/{document_id}", response_model=DocumentStatus)
    def get_document(document_id: str, owner_id: UUID = Depends(current_user)) -> DocumentStatus:
        row = active_persistence.get_document(validate_document_id(document_id), owner_id)
        if not row:
            raise HTTPException(status_code=404, detail="Document does not exist")
        return DocumentStatus(
            document_id=row["id"],
            status=row["status"],
            original_filename=row["original_filename"],
            counts={
                "pages": row.get("page_count", 0),
                "pictures": row.get("figure_count", 0),
                "tables": row.get("table_count", 0),
                "formulas": row.get("formula_count", 0),
            },
            warnings=row.get("warnings") or [],
            errors=row.get("errors") or [],
            chunk_count=row.get("chunk_count") or 0,
            embedding_status=row.get("embedding_status", "not_started"),
        )

    @app.post("/v1/documents/{document_id}/retrieve", response_model=RetrievalResponse)
    def retrieve_document(
        document_id: str,
        request: RetrievalRequest,
        owner_id: UUID = Depends(current_user),
    ) -> RetrievalResponse:
        safe_document_id = validate_document_id(document_id)
        document = active_persistence.get_document(safe_document_id, owner_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document does not exist")
        if document.get("status") != "ready":
            raise HTTPException(status_code=409, detail="Document parsing has not completed")
        if document.get("embedding_status", "not_started") != "ready":
            raise HTTPException(status_code=409, detail="Document retrieval index is not ready")
        try:
            active_retriever = retriever or HybridRetriever(
                active_persistence, OpenAIEmbedder(EmbeddingSettings.from_env())
            )
            with trace(
                "api.retrieve_document",
                run_type="chain",
                inputs={"document_id": safe_document_id, "top_k": request.top_k},
                tags=["saral", "api", "retrieval"],
            ):
                return active_retriever.retrieve(
                    safe_document_id, owner_id, request.question, request.top_k
                )
        except ConfigurationError as exc:
            raise HTTPException(
                status_code=503, detail="Retrieval provider is not configured"
            ) from exc
        except RetrievalError as exc:
            raise HTTPException(status_code=502, detail="Document retrieval is unavailable") from exc

    @app.get("/v1/documents/{document_id}/assets/{asset_id}")
    def get_asset(
        document_id: str, asset_id: str, owner_id: UUID = Depends(current_user)
    ) -> Response:
        try:
            payload, media_type = active_persistence.download_asset(
                validate_document_id(document_id), asset_id, owner_id
            )
        except PersistenceError as exc:
            raise HTTPException(status_code=404, detail="Asset does not exist") from exc
        return Response(content=payload, media_type=media_type)

    return app


def _supabase_user_resolver(persistence: SupabasePersistence) -> UserResolver:
    """Verify browser JWTs server-side; service-role credentials never leave Python."""

    def resolve(authorization: Optional[str]) -> UUID:
        if authorization and authorization.startswith("Bearer "):
            response = persistence._client.auth.get_user(authorization[7:])
            return UUID(str(response.user.id))
        dev_user = os.environ.get("SARAL_DEV_USER_ID")
        if os.environ.get("SARAL_ENVIRONMENT") == "development" and dev_user:
            return UUID(dev_user)
        raise ValueError("missing bearer token")

    return resolve


app = create_app() if os.environ.get("SARAL_EAGER_APP") == "1" else None
