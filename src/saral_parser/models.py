"""Validated public configuration and result models."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class OcrEngine(str, Enum):
    """OCR implementations supported by the installed Docling environment."""

    AUTO = "auto"
    OCRMAC = "ocrmac"
    TESSERACT_CLI = "tesseract_cli"


class PictureDescriptionMode(str, Enum):
    """Privacy-aware picture description modes."""

    DISABLED = "disabled"
    SMOLVLM_LOCAL = "smolvlm_local"
    GRANITE_LOCAL = "granite_local"


class ParseOptions(BaseModel):
    """Request-scoped pipeline configuration with conservative defaults."""

    model_config = ConfigDict(extra="forbid")

    enable_ocr: bool = False
    ocr_engine: OcrEngine = OcrEngine.AUTO
    ocr_languages: List[str] = Field(default_factory=lambda: ["en"], max_length=8)
    extract_table_structure: bool = True
    enable_formula_enrichment: bool = False
    enable_picture_classification: bool = False
    picture_description_mode: PictureDescriptionMode = PictureDescriptionMode.DISABLED
    generate_page_images: bool = True
    generate_picture_images: bool = True
    image_scale: float = Field(default=1.5, ge=1.0, le=3.0)
    timeout_seconds: int = Field(default=900, ge=30, le=3600)
    debug_full_document_logging: bool = False

    @field_validator("ocr_languages")
    @classmethod
    def validate_ocr_languages(cls, value: List[str]) -> List[str]:
        if not value or any(not language.strip() or len(language) > 35 for language in value):
            raise ValueError("ocr_languages must contain non-empty language tags")
        return value


class ParserSettings(BaseModel):
    """Process-level constraints and locations; paths stay inside ``workspace_root``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    workspace_root: Path
    paper_folder: Path
    upload_folder: Path
    output_folder: Path
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, ge=1)
    allowed_extensions: frozenset[str] = frozenset({".pdf"})
    allowed_origins: frozenset[str] = frozenset(
        {
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        }
    )
    preview_characters: int = Field(default=600, ge=80, le=10_000)

    @classmethod
    def from_workspace(cls, workspace_root: Path) -> "ParserSettings":
        root = workspace_root.resolve()
        allowed_origins = frozenset(
            origin.strip()
            for origin in os.environ.get(
                "SARAL_ALLOWED_ORIGINS",
                (
                    "http://localhost:5173,http://127.0.0.1:5173,"
                    "http://localhost:8080,http://127.0.0.1:8080"
                ),
            ).split(",")
            if origin.strip()
        )
        return cls(
            workspace_root=root,
            paper_folder=root / "paper_folder",
            upload_folder=root / "uploads",
            output_folder=root / "outputs",
            allowed_origins=allowed_origins,
        )


class InputMetadata(BaseModel):
    """Identity and safety-relevant facts for a submitted source file."""

    document_id: str
    content_sha256: str
    original_filename: str
    stored_filename: str
    media_type: str
    size_bytes: int
    source_path: str


class ArtifactManifest(BaseModel):
    """Stable, parser-only handoff contract for a completed document job."""

    document_id: str
    content_sha256: str
    input: InputMetadata
    options: ParseOptions
    status: str
    created_at: str
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    counts: Dict[str, int] = Field(default_factory=dict)
    artifacts: Dict[str, str] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    notes: Dict[str, Any] = Field(default_factory=dict)


class ChunkOptions(BaseModel):
    """Deterministic HybridChunker configuration saved with every chunking job."""

    model_config = ConfigDict(extra="forbid")

    tokenizer_model: str = "BAAI/bge-small-en-v1.5"
    max_tokens: int = Field(default=512, ge=64, le=8192)
    merge_peers: bool = True


class DocumentChunk(BaseModel):
    """Ordered, grounded chunk ready for later embedding."""

    chunk_id: str
    document_id: str
    chunk_index: int = Field(ge=0)
    text: str
    contextualized_text: str
    headings: List[str] = Field(default_factory=list)
    captions: List[str] = Field(default_factory=list)
    source_refs: List[str] = Field(min_length=1)
    page_numbers: List[int] = Field(default_factory=list)
    provenance: List[Dict[str, Any]] = Field(default_factory=list)
    asset_ids: List[str] = Field(default_factory=list)
    content_types: List[str] = Field(default_factory=list)
    token_count: int = Field(ge=1)
    chunker_version: str


class ChunkingResult(BaseModel):
    """Validated ordered output of one chunking run."""

    chunks: List[DocumentChunk]
    warnings: List[str] = Field(default_factory=list)
    options: ChunkOptions


class SupabaseSettings(BaseModel):
    """Server-only Supabase and Redis settings; secrets never enter API responses."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str
    service_role_key: SecretStr
    bucket: str = "saral-documents"
    redis_url: SecretStr = SecretStr("redis://127.0.0.1:6379/0")
    request_timeout_seconds: int = Field(default=30, ge=5, le=120)
    celery_visibility_timeout_seconds: int = Field(default=3600, ge=900, le=86_400)

    @classmethod
    def from_env(cls) -> "SupabaseSettings":
        """Load production persistence settings without logging secret values."""
        url = os.environ.get("SUPABASE_URL", "").strip()
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        if not url or not key:
            from .exceptions import ConfigurationError

            raise ConfigurationError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
        return cls(
            url=url,
            service_role_key=key,
            bucket=os.environ.get("SARAL_STORAGE_BUCKET", "saral-documents"),
            redis_url=os.environ.get("SARAL_REDIS_URL", "redis://127.0.0.1:6379/0"),
        )


class EmbeddingSettings(BaseModel):
    """Server-only OpenAI embedding configuration for one retrieval index."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: SecretStr
    model: str = "text-embedding-3-small"
    dimensions: int = Field(default=1536, ge=1, le=2000)
    batch_size: int = Field(default=64, ge=1, le=2048)
    timeout_seconds: float = Field(default=30.0, ge=5.0, le=120.0)
    max_input_tokens: int = Field(default=8191, ge=1, le=8191)

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        """Load the embedding provider configuration without logging its API key."""
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            from .exceptions import ConfigurationError

            raise ConfigurationError("OPENAI_API_KEY is required for retrieval")
        return cls(
            api_key=SecretStr(key),
            model=os.environ.get("SARAL_EMBEDDING_MODEL", "text-embedding-3-small"),
            dimensions=int(os.environ.get("SARAL_EMBEDDING_DIMENSIONS", "1536")),
            batch_size=int(os.environ.get("SARAL_EMBEDDING_BATCH_SIZE", "64")),
            timeout_seconds=float(os.environ.get("SARAL_EMBEDDING_TIMEOUT_SECONDS", "30")),
        )


class AcceptedDocument(BaseModel):
    """HTTP 202 response returned after the durable job has been dispatched."""

    document_id: str
    job_id: UUID
    status: str = "queued"
    status_url: str


class DocumentStatus(BaseModel):
    """Small polling response backed by Postgres rather than artifact files."""

    document_id: str
    status: str
    original_filename: str
    counts: Dict[str, int] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    chunk_count: int = 0
    embedding_status: str = "not_started"


class RetrievalRequest(BaseModel):
    """Validated request for owner-scoped hybrid document retrieval."""

    question: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=6, ge=5, le=8)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("question must not be blank")
        return question


class RetrievedChunk(BaseModel):
    """One fused retrieval hit with source provenance retained for citations."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    contextualized_text: str
    headings: List[str] = Field(default_factory=list)
    captions: List[str] = Field(default_factory=list)
    source_refs: List[str] = Field(default_factory=list)
    page_numbers: List[int] = Field(default_factory=list)
    provenance: List[Dict[str, Any]] = Field(default_factory=list)
    asset_ids: List[str] = Field(default_factory=list)
    content_types: List[str] = Field(default_factory=list)
    rrf_score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None


class RetrievalResponse(BaseModel):
    """Hybrid-search results ready for a grounded generation prompt."""

    document_id: str
    chunks: List[RetrievedChunk]

