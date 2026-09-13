"""Owner-scoped dense, sparse, and reciprocal-rank-fused retrieval."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence
from uuid import UUID

from .embeddings import OpenAIEmbedder, vector_literal
from .exceptions import RetrievalError
from .models import RetrievedChunk, RetrievalResponse
from .persistence import SupabasePersistence

LOGGER = logging.getLogger(__name__)
_CANDIDATE_LIMIT = 20
_RRF_K = 60


class HybridRetriever:
    """Retrieve one document through parallel dense and sparse searches."""

    def __init__(self, persistence: SupabasePersistence, embedder: OpenAIEmbedder) -> None:
        self.persistence = persistence
        self.embedder = embedder

    def retrieve(
        self, document_id: str, owner_id: UUID, question: str, top_k: int = 6
    ) -> RetrievalResponse:
        """Return deterministic fused chunks with all parser provenance intact."""
        started = time.monotonic()
        try:
            embedding = vector_literal(self.embedder.embed_query(question))
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="saral-retrieval") as executor:
                dense = executor.submit(
                    self.persistence.search_document_chunks_dense,
                    document_id,
                    owner_id,
                    embedding,
                    _CANDIDATE_LIMIT,
                )
                sparse = executor.submit(
                    self.persistence.search_document_chunks_sparse,
                    document_id,
                    owner_id,
                    question,
                    _CANDIDATE_LIMIT,
                )
                dense_rows = dense.result()
                sparse_rows = sparse.result()
        except Exception as exc:
            if isinstance(exc, RetrievalError):
                raise
            raise RetrievalError("Document retrieval is temporarily unavailable") from exc

        chunks = fuse_rankings(dense_rows, sparse_rows, top_k=top_k)
        LOGGER.info(
            "Hybrid retrieval completed",
            extra={
                "document_id": document_id,
                "owner_id": str(owner_id),
                "dense_count": len(dense_rows),
                "sparse_count": len(sparse_rows),
                "result_count": len(chunks),
                "duration_ms": round((time.monotonic() - started) * 1000),
            },
        )
        return RetrievalResponse(document_id=document_id, chunks=chunks)


def fuse_rankings(
    dense_rows: Sequence[dict[str, Any]],
    sparse_rows: Sequence[dict[str, Any]],
    *,
    top_k: int,
    rrf_k: int = _RRF_K,
) -> list[RetrievedChunk]:
    """Fuse ranked rows with RRF while preserving each chunk's citation metadata."""
    if top_k < 1:
        raise RetrievalError("top_k must be positive")
    if rrf_k < 1:
        raise RetrievalError("rrf_k must be positive")

    combined: dict[str, dict[str, Any]] = {}
    for rows, rank_name in ((dense_rows, "dense_rank"), (sparse_rows, "sparse_rank")):
        for rank, row in enumerate(rows, start=1):
            chunk_id = str(row.get("chunk_id") or row.get("id") or "")
            if not chunk_id:
                continue
            entry = combined.setdefault(chunk_id, {**row, "rrf_score": 0.0})
            entry[rank_name] = rank
            entry["rrf_score"] += 1.0 / (rrf_k + rank)

    ordered = sorted(
        combined.values(),
        key=lambda item: (
            -float(item["rrf_score"]),
            min(item.get("dense_rank") or 10**9, item.get("sparse_rank") or 10**9),
            int(item.get("chunk_index", 10**9)),
            str(item.get("chunk_id") or item.get("id")),
        ),
    )
    return [_to_retrieved_chunk(item) for item in ordered[:top_k]]


def _to_retrieved_chunk(row: dict[str, Any]) -> RetrievedChunk:
    chunk_id = str(row.get("chunk_id") or row.get("id") or "")
    if not chunk_id or not row.get("document_id"):
        raise RetrievalError("Search result is missing a chunk identity")
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=str(row["document_id"]),
        chunk_index=int(row.get("chunk_index", 0)),
        text=str(row.get("text") or ""),
        contextualized_text=str(row.get("contextualized_text") or row.get("text") or ""),
        headings=list(row.get("headings") or []),
        captions=list(row.get("captions") or []),
        source_refs=list(row.get("source_refs") or []),
        page_numbers=list(row.get("page_numbers") or []),
        provenance=list(row.get("provenance") or []),
        asset_ids=list(row.get("asset_ids") or []),
        content_types=list(row.get("content_types") or []),
        rrf_score=round(float(row["rrf_score"]), 12),
        dense_rank=row.get("dense_rank"),
        sparse_rank=row.get("sparse_rank"),
    )
