"""Owner-scoped dense, sparse, and reciprocal-rank-fused retrieval."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence
from uuid import UUID

from .embeddings import OpenRouterEmbedder, LocalMathEmbedder, vector_literal
from .exceptions import RetrievalError
from .models import RetrievedChunk, RetrievalResponse
from .persistence import SupabasePersistence

LOGGER = logging.getLogger(__name__)
_CANDIDATE_LIMIT = 20
_RRF_K = 60

ROUTER_PROMPT = """
You are an intelligent intent analyzer for a scientific paper RAG system.
Classify the user's query into one of three exact categories:
1. "text_only": The user wants a conceptual explanation without needing specific mathematical equations.
2. "formula_and_text": The user is asking about a specific calculation, equation, or mathematical concept.
3. "all_formulas": The user is exhaustively asking for a list of all formulas/equations in the paper.

User Query: "{query}"

Output ONLY valid JSON matching this schema: {{"intent": "<category>"}}
"""


class HybridRetriever:
    """Retrieve one document through parallel dense and sparse searches."""

    def __init__(self, persistence: SupabasePersistence, embedder: OpenRouterEmbedder, llm_client: Any | None = None) -> None:
        self.persistence = persistence
        self.embedder = embedder
        self.llm_client = llm_client
        if self.llm_client is None:
            import os
            from openai import OpenAI
            api_key = os.environ.get("OPENROUTER_API_KEY")
            base_url = os.environ.get("SARAL_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            if api_key:
                self.llm_client = OpenAI(api_key=api_key, base_url=base_url, max_retries=2)

    def retrieve(
        self, document_id: str, owner_id: UUID, question: str, top_k: int = 6
    ) -> RetrievalResponse:
        """Return deterministic fused chunks with all parser provenance intact."""
        import os
        import json
        started = time.monotonic()
        
        dense_rows: list[dict[str, Any]] = []
        sparse_rows: list[dict[str, Any]] = []
        math_rows: list[dict[str, Any]] = []
        
        # 1. Router Phase
        intent = "formula_and_text"
        if self.llm_client:
            try:
                completion = self.llm_client.chat.completions.create(
                    model=os.environ.get("SARAL_GENERATION_MODEL", "openai/gpt-4.1-mini"),
                    messages=[{"role": "system", "content": ROUTER_PROMPT.format(query=question)}],
                    response_format={"type": "json_object"},
                    max_tokens=50,
                    temperature=0.0
                )
                content = completion.choices[0].message.content
                if content:
                    intent_json = json.loads(content)
                    intent = intent_json.get("intent", "formula_and_text")
                    LOGGER.info(f"Router classified query intent as: {intent}")
            except Exception as exc:
                LOGGER.warning("Router classification failed, defaulting to multimodal", exc_info=exc)

        # 2. Retrieval Phase
        try:
            embedding_text = vector_literal(self.embedder.embed_query(question))
            
            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="saral-retrieval") as executor:
                dense_text = executor.submit(
                    self.persistence.search_document_chunks_dense,
                    document_id,
                    owner_id,
                    embedding_text,
                    _CANDIDATE_LIMIT,
                )
                sparse = executor.submit(
                    self.persistence.search_document_chunks_sparse,
                    document_id,
                    owner_id,
                    question,
                    _CANDIDATE_LIMIT,
                )
                
                if intent in ("formula_and_text", "all_formulas"):
                    math_embedder = LocalMathEmbedder(self.embedder.settings)
                    embedding_math = vector_literal(math_embedder.embed_query(question))
                    dense_math = executor.submit(
                        self.persistence.search_document_chunks_dense,
                        document_id,
                        owner_id,
                        embedding_math,
                        _CANDIDATE_LIMIT,
                    )
                else:
                    dense_math = None

                dense_rows = dense_text.result()
                sparse_rows = sparse.result()
                math_rows = dense_math.result() if dense_math else []
        except Exception as exc:
            if isinstance(exc, RetrievalError):
                raise
            raise RetrievalError("Document retrieval is temporarily unavailable") from exc

        # 3. Filtering Phase
        if intent == "text_only":
            dense_rows = [row for row in dense_rows if "formula" not in row.get("content_types", [])]
            sparse_rows = [row for row in sparse_rows if "formula" not in row.get("content_types", [])]

        chunks = fuse_rankings(dense_rows, sparse_rows, math_rows, top_k=top_k)
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
    *rankings_lists: Sequence[dict[str, Any]],
    top_k: int,
    rrf_k: int = _RRF_K,
) -> list[RetrievedChunk]:
    """Fuse N ranked rows with RRF while preserving each chunk's citation metadata."""
    if top_k < 1:
        raise RetrievalError("top_k must be positive")
    if rrf_k < 1:
        raise RetrievalError("rrf_k must be positive")

    combined: dict[str, dict[str, Any]] = {}
    
    for row_list in rankings_lists:
        for rank, row in enumerate(row_list, start=1):
            chunk_id = str(row.get("chunk_id") or row.get("id") or "")
            if not chunk_id:
                continue
            entry = combined.setdefault(chunk_id, {**row, "rrf_score": 0.0, "dense_rank": None, "sparse_rank": None})
            
            # Keep track of minimum observed rank for secondary sorting
            if entry.get("min_rank") is None or rank < entry["min_rank"]:
                entry["min_rank"] = rank
                
            entry["rrf_score"] += (1.0 / (rrf_k + rank))

    ordered = sorted(
        combined.values(),
        key=lambda item: (
            -float(item["rrf_score"]),
            item.get("min_rank") or 10**9,
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
