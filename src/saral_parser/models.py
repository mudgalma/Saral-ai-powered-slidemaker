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
    """Server-only OpenRouter embedding configuration for one retrieval index."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: SecretStr
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openai/text-embedding-3-small"
    dimensions: int = Field(default=1536, ge=1, le=2000)
    batch_size: int = Field(default=64, ge=1, le=2048)
    timeout_seconds: float = Field(default=30.0, ge=5.0, le=120.0)
    max_input_tokens: int = Field(default=8191, ge=1, le=8191)

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        """Load the embedding provider configuration without logging its API key."""
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            from .exceptions import ConfigurationError

            raise ConfigurationError("OPENROUTER_API_KEY is required for retrieval")
        return cls(
            api_key=SecretStr(key),
            base_url=os.environ.get("SARAL_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            model=os.environ.get("SARAL_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
            dimensions=int(os.environ.get("SARAL_EMBEDDING_DIMENSIONS", "1536")),
            batch_size=int(os.environ.get("SARAL_EMBEDDING_BATCH_SIZE", "64")),
            timeout_seconds=float(os.environ.get("SARAL_EMBEDDING_TIMEOUT_SECONDS", "30")),
        )


class GenerationSettings(BaseModel):
    """Server-only OpenRouter configuration for bounded grounded generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: SecretStr
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openai/gpt-4.1-mini"
    timeout_seconds: float = Field(default=120.0, ge=5.0, le=240.0)
    max_output_tokens: int = Field(default=4000, ge=100, le=8000)

    @classmethod
    def from_env(cls) -> "GenerationSettings":
        """Load generation configuration without logging its API key."""
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            from .exceptions import ConfigurationError

            raise ConfigurationError("OPENROUTER_API_KEY is required for generation")
        return cls(
            api_key=SecretStr(key),
            base_url=os.environ.get("SARAL_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            model=os.environ.get("SARAL_GENERATION_MODEL", "openai/gpt-4.1-mini"),
            timeout_seconds=float(os.environ.get("SARAL_GENERATION_TIMEOUT_SECONDS", "120")),
            max_output_tokens=int(os.environ.get("SARAL_GENERATION_MAX_OUTPUT_TOKENS", "4000")),
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


class ArtifactType(str, Enum):
    """Supported grounded artifacts returned by the generation API."""

    ANSWER = "answer"
    SUMMARY = "summary"
    SCRIPT = "script"
    SLIDE_OUTLINE = "slide_outline"
    TWEET_THREAD = "tweet_thread"
    LINKEDIN_POST = "linkedin_post"


class GenerationLength(str, Enum):
    """Output budgets exposed to API callers."""

    BRIEF = "brief"
    STANDARD = "standard"
    EXTENDED = "extended"


class GenerationRequest(BaseModel):
    """Validated instructions for a document-grounded generated artifact."""

    model_config = ConfigDict(extra="forbid")

    artifact_type: ArtifactType = ArtifactType.ANSWER
    audience: str = Field(default="general audience", min_length=1, max_length=120)
    length: GenerationLength = GenerationLength.STANDARD
    style: str = Field(default="plain English", min_length=1, max_length=120)
    user_instruction: str = Field(min_length=1, max_length=4_000)
    slide_count: Optional[int] = Field(default=None, ge=2, le=20)

    @field_validator("audience", "style", "user_instruction")
    @classmethod
    def validate_generation_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("generation fields must not be blank")
        return cleaned


class EvidenceChunk(BaseModel):
    """Citation-safe projection of one retrieved chunk for the LLM prompt."""

    chunk_id: str
    text: str = Field(min_length=1)
    page_numbers: List[int] = Field(default_factory=list)
    heading: Optional[str] = None
    provenance: List[Dict[str, Any]] = Field(default_factory=list)


class EvidenceAsset(BaseModel):
    """One source visual linked to a retrieved chunk and safe for UI retrieval."""

    asset_id: str = Field(min_length=1)
    source_chunk_id: str = Field(min_length=1)
    page_numbers: List[int] = Field(default_factory=list)
    caption: Optional[str] = Field(default=None, max_length=2_000)


class EvidencePack(BaseModel):
    """Bounded evidence supplied to one grounded generation workflow."""

    document_id: str
    chunks: List[EvidenceChunk] = Field(min_length=1, max_length=8)
    assets: List[EvidenceAsset] = Field(default_factory=list, max_length=12)


class GroundedClaim(BaseModel):
    """A generated factual claim and the chunks offered as its evidence."""

    text: str = Field(min_length=1, max_length=1_500)
    citation_ids: List[str] = Field(min_length=1, max_length=4)


class GeneratedArtifactDraft(BaseModel):
    """Schema parsed directly from the LLM before deterministic grounding checks."""

    title: str = Field(min_length=1, max_length=180)
    content: str = Field(min_length=1, max_length=16_000)
    claims: List[GroundedClaim] = Field(min_length=1, max_length=50)


class SlideProvenance(BaseModel):
    """One factual slide claim and the retrieved chunks that support it."""

    claim: str = Field(min_length=1, max_length=1_500)
    citation_ids: List[str] = Field(min_length=1, max_length=4)


class SlideDraft(BaseModel):
    """A renderer-ready slide generated from bounded document evidence."""

    slide_number: int = Field(ge=1, le=20)
    role: str = Field(min_length=1, max_length=80)
    header_takeaway: str = Field(min_length=1, max_length=180)
    bullets: List[str] = Field(min_length=1, max_length=6)
    speaker_notes: List[str] = Field(min_length=3, max_length=3)
    spoken_script: str = Field(min_length=1, max_length=2_500)
    provenance: List[SlideProvenance] = Field(min_length=1, max_length=20)


class SlideDeckDraft(BaseModel):
    """Structured slide-deck output before deterministic grounding validation."""

    title: str = Field(min_length=1, max_length=180)
    slides: List[SlideDraft] = Field(min_length=2, max_length=20)


class ArtifactCitation(BaseModel):
    """Display-ready citation derived only from retrieved evidence metadata."""

    chunk_id: str
    page_numbers: List[int] = Field(default_factory=list)
    heading: Optional[str] = None


class ArtifactVisualAsset(BaseModel):
    """A UI-displayable source visual selected alongside grounded evidence."""

    asset_id: str = Field(min_length=1)
    source_chunk_id: str = Field(min_length=1)
    page_numbers: List[int] = Field(default_factory=list)
    caption: Optional[str] = Field(default=None, max_length=2_000)


class GroundingReport(BaseModel):
    """Deterministic audit result for one generated draft."""

    passed: bool
    issues: List[str] = Field(default_factory=list)
    cited_chunk_ids: List[str] = Field(default_factory=list)


class GeneratedArtifact(BaseModel):
    """A validated artifact that may be presented to an API client."""

    artifact_type: ArtifactType
    title: str
    content: str
    citations: List[ArtifactCitation] = Field(min_length=1)
    visual_assets: List[ArtifactVisualAsset] = Field(default_factory=list, max_length=12)
    deck: Optional[SlideDeckDraft] = None


class GenerationResponse(BaseModel):
    """Grounded generation result, including a safe flagged outcome."""

    document_id: str
    status: str
    artifact: Optional[GeneratedArtifact] = None
    grounding: GroundingReport
    attempts: int = Field(ge=0, le=2)


class ConversationBranch(str, Enum):
    """Bounded routes supported by the Phase 2 conversation workflow."""

    NEW_GENERATION = "new_generation"
    REVISION = "revision"
    QUESTION = "question"


class ConversationMessageRequest(BaseModel):
    """Validated user turn submitted to one document-bound conversation thread."""

    model_config = ConfigDict(extra="forbid")

    thread_id: UUID
    message: str = Field(min_length=1, max_length=2_000)
    audience: Optional[str] = Field(default=None, min_length=1, max_length=120)
    length: Optional[GenerationLength] = None
    style: Optional[str] = Field(default=None, min_length=1, max_length=120)

    @field_validator("message", "audience", "style")
    @classmethod
    def validate_message(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("conversation text fields must not be blank")
        return cleaned


class ConversationIntent(BaseModel):
    """Resolved artifact requirements and route for a single user message."""

    branch: ConversationBranch
    artifact_type: ArtifactType
    audience: str = Field(min_length=1, max_length=120)
    length: GenerationLength
    style: str = Field(min_length=1, max_length=120)
    slide_count: Optional[int] = Field(default=None, ge=2, le=20)
    is_revision: bool = False


class ConversationMessage(BaseModel):
    """A bounded persisted conversation message, never a prompt transcript."""

    id: UUID
    role: str
    content: str = Field(min_length=1, max_length=16_000)
    created_at: str


class ArtifactVersion(BaseModel):
    """Immutable version of a grounded artifact with optional parent and delta."""

    id: UUID
    version_number: int = Field(ge=1)
    parent_version_id: Optional[UUID] = None
    artifact: GeneratedArtifact
    delta: Optional[str] = Field(default=None, max_length=8_000)
    created_at: str


class ConversationState(BaseModel):
    """Persistent raw state used to resolve a document conversation turn."""

    thread_id: UUID
    document_id: str
    messages: List[ConversationMessage] = Field(default_factory=list, max_length=40)
    current_artifact: Optional[ArtifactVersion] = None
    previous_versions: List[ArtifactVersion] = Field(default_factory=list, max_length=20)


class ConversationResponse(BaseModel):
    """Outcome of a routed, grounded document conversation turn."""

    thread_id: UUID
    document_id: str
    intent: ConversationIntent
    generation: GenerationResponse
    version: Optional[ArtifactVersion] = None
