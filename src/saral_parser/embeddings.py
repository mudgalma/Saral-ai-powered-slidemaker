"""Bounded OpenRouter embedding client used by indexing and retrieval."""

from __future__ import annotations

import json
import time
from typing import Any, Sequence

from .exceptions import EmbeddingError
from .models import EmbeddingSettings

_TRANSIENT_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


class OpenRouterEmbedder:
    """Create fixed-size OpenRouter vectors without exposing provider failures."""

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
            base_url=self.settings.base_url,
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


class LocalMathEmbedder:
    """Create local MathBERTa vectors for formula chunks with zero-padding."""

    def __init__(self, settings: EmbeddingSettings, model_name: str = "witiko/mathberta") -> None:
        self.settings = settings
        self.model_name = model_name
        self._model = None

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device="cpu")
            except ImportError as exc:
                raise EmbeddingError("sentence-transformers is not installed") from exc
            except Exception as exc:
                raise EmbeddingError(f"Could not load local math model {self.model_name}") from exc
        return self._model

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        
        model = self._load_model()
        try:
            embeddings = model.encode(texts, convert_to_numpy=True)
            # Pad or truncate to match the expected database dimensions (e.g. 1536 for OpenAI)
            target_dim = self.settings.dimensions
            padded_embeddings = []
            for emb in embeddings:
                emb_list = emb.tolist()
                if len(emb_list) < target_dim:
                    emb_list.extend([0.0] * (target_dim - len(emb_list)))
                elif len(emb_list) > target_dim:
                    emb_list = emb_list[:target_dim]
                padded_embeddings.append(emb_list)
            return padded_embeddings
        except Exception as exc:
            raise EmbeddingError("Failed to encode text with local math model") from exc

