"""Domain-specific failures exposed by the parsing service."""


class ParserError(Exception):
    """Base class for expected parser failures."""


class InputValidationError(ParserError):
    """Raised when an input is unsafe, unsupported, or outside configured limits."""


class ConversionError(ParserError):
    """Raised when Docling cannot convert a validated document."""


class ArtifactError(ParserError):
    """Raised when a conversion artifact cannot be written or verified."""


class ConfigurationError(ParserError):
    """Raised when required process configuration is absent or unsafe."""


class PersistenceError(ParserError):
    """Raised when Supabase persistence cannot complete safely."""


class ChunkingError(ParserError):
    """Raised when a validated Docling document cannot be chunked."""


class JobDispatchError(ParserError):
    """Raised when a parsing job cannot be queued in Redis."""


class EmbeddingError(ParserError):
    """Raised when an embedding provider cannot create a safe vector."""


class RetrievalError(ParserError):
    """Raised when hybrid retrieval cannot return a safe result."""

