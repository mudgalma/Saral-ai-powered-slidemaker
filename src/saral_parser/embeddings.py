"""Bounded OpenAI embedding client used by indexing and retrieval."""

from __future__ import annotations

import json
import time
from typing import Any, Sequence

from .exceptions import EmbeddingError
from .models import EmbeddingSettings

_TRANSIENT_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


class OpenAIEmbedder:
    """Create fixed-size OpenAI vectors without exposing provider failures."""

    def __init__(self, settings: EmbeddingSettings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client or self._create_client()

    def embed_query(self, question: str) -> list[float]:
        """Embed one validated user question."""
        return self.embed_texts([question])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ordered text inputs in bounded batches and preserve their order."""
        normalized = [self._validate_text(text) for text in texts]
        if not normalized:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(normalized), self.settings.batch_size):
            batch = normalized[start : start + self.settings.batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _create_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - installation boundary
            raise EmbeddingError("The pinned openai package is not installed") from exc
        return OpenAI(
            api_key=self.settings.api_key.get_secret_value(),
            timeout=self.settings.timeout_seconds,
            max_retries=0,
        )

    def _validate_text(self, text: str) -> str:
        normalized = text.strip()
        if not normalized:
            raise EmbeddingError("Embedding text must not be empty")
        # A three-byte-per-token estimate is deliberately conservative and is local;
        # it avoids a tokenizer vocabulary download in workers and tests.
        estimated_tokens = (len(normalized.encode("utf-8")) + 2) // 3
        if estimated_tokens > self.settings.max_input_tokens:
            raise EmbeddingError("Embedding text exceeds the configured token limit")
        return normalized

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(3):
            try:
                response = self._client.embeddings.create(
                    model=self.settings.model,
                    input=batch,
                    dimensions=self.settings.dimensions,
                )
                values = sorted(response.data, key=lambda item: item.index)
                vectors = [list(item.embedding) for item in values]
                if len(vectors) != len(batch) or any(
                    len(vector) != self.settings.dimensions for vector in vectors
                ):
                    raise EmbeddingError("Embedding provider returned an invalid vector shape")
                return vectors
            except EmbeddingError:
                raise
            except Exception as exc:
                if attempt == 2 or not _is_transient(exc):
                    raise EmbeddingError("Embedding provider request failed") from exc
                time.sleep(0.25 * (2**attempt))
        raise AssertionError("unreachable")


def vector_literal(vector: Sequence[float]) -> str:
    """Encode one validated vector for the pgvector RPC argument."""
    if not vector:
        raise EmbeddingError("Embedding vector must not be empty")
    return json.dumps(list(vector), separators=(",", ":"))


def _is_transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    try:
        if int(status) in _TRANSIENT_STATUS_CODES:
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(exc, (ConnectionError, TimeoutError))
