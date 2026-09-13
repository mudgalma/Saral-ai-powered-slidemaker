"""Docling-native hybrid chunking with explicit grounding invariants."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from langsmith import traceable

from .exceptions import ChunkingError
from .models import ChunkingResult, ChunkOptions, DocumentChunk

CHUNKER_VERSION = "docling-hybrid-v1"


class HybridDocumentChunker:
    """Chunk a DoclingDocument by hierarchy, then fit chunks to a tokenizer budget."""

    def __init__(self, options: ChunkOptions, chunker: Any | None = None) -> None:
        self.options = options
        self._chunker = chunker or self._create_chunker(options)

    @staticmethod
    def _create_chunker(options: ChunkOptions) -> Any:
        try:
            from docling_core.transforms.chunker import HybridChunker
            from docling_core.transforms.chunker.tokenizer.huggingface import (
                HuggingFaceTokenizer,
            )
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise ChunkingError("Docling HybridChunker dependencies are unavailable") from exc

        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(options.tokenizer_model),
            max_tokens=options.max_tokens,
        )
        return HybridChunker(tokenizer=tokenizer, merge_peers=options.merge_peers)

    @traceable(
        name="chunking.load_and_run",
        run_type="tool",
        process_inputs=lambda inputs: {"document_id": inputs.get("document_id")},
        process_outputs=lambda output: {
            "chunk_count": len(output.chunks),
            "warning_count": len(output.warnings),
        },
        tags=["saral", "chunking", "hybrid"],
    )
    def chunk_path(self, document_id: str, document_json: Path) -> ChunkingResult:
        """Load authoritative Docling JSON and return validated ordered chunks."""
        try:
            from docling_core.types.doc import DoclingDocument

            document = DoclingDocument.model_validate_json(
                document_json.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise ChunkingError("Could not load authoritative Docling JSON") from exc
        return self.chunk_document(document_id, document)

    @traceable(
        name="chunking.build_and_validate",
        run_type="tool",
        process_inputs=lambda inputs: {"document_id": inputs.get("document_id")},
        process_outputs=lambda output: {
            "chunk_count": len(output.chunks),
            "formula_chunk_count": sum("formula" in chunk.content_types for chunk in output.chunks),
            "table_chunk_count": sum("table" in chunk.content_types for chunk in output.chunks),
            "max_token_count": max((chunk.token_count for chunk in output.chunks), default=0),
            "warning_count": len(output.warnings),
        },
        tags=["saral", "chunking", "validation"],
    )
    def chunk_document(self, document_id: str, document: Any) -> ChunkingResult:
        chunks: List[DocumentChunk] = []
        for index, raw_chunk in enumerate(self._chunker.chunk(document)):
            meta = raw_chunk.meta
            items = list(getattr(meta, "doc_items", []) or [])
            refs = _unique(_self_ref(item) for item in items if _self_ref(item))
            if not refs:
                raise ChunkingError(f"Chunk {index} has no Docling source references")
            provenance = _provenance(items)
            contextualized = self._chunker.contextualize(raw_chunk)
            token_count = self._count_tokens(contextualized)
            labels = _unique(str(getattr(item, "label", "unknown")) for item in items)
            assets = [_asset_id(document_id, ref) for ref in refs if _asset_kind(ref)]
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document_id}:{CHUNKER_VERSION}:{index:06d}",
                    document_id=document_id,
                    chunk_index=index,
                    text=raw_chunk.text,
                    contextualized_text=contextualized,
                    headings=list(getattr(meta, "headings", []) or []),
                    captions=list(getattr(meta, "captions", []) or []),
                    source_refs=refs,
                    page_numbers=sorted(
                        {int(item["page_number"]) for item in provenance if item.get("page_number")}
                    ),
                    provenance=provenance,
                    asset_ids=assets,
                    content_types=labels,
                    token_count=token_count,
                    chunker_version=CHUNKER_VERSION,
                )
            )
        warnings = self._validate(document, chunks)
        return ChunkingResult(chunks=chunks, warnings=warnings, options=self.options)

    def _count_tokens(self, text: str) -> int:
        tokenizer = getattr(self._chunker, "tokenizer", None)
        if tokenizer and hasattr(tokenizer, "count_tokens"):
            return int(tokenizer.count_tokens(text))
        return max(1, len(text.split()))  # test-double fallback only

    def _validate(self, document: Any, chunks: Sequence[DocumentChunk]) -> List[str]:
        if not chunks:
            raise ChunkingError("Hybrid chunking produced no chunks")
        ordered_refs = [
            _self_ref(item) for item, _level in document.iterate_items() if _self_ref(item)
        ]
        positions = {ref: index for index, ref in enumerate(ordered_refs)}
        starts = [
            min(positions[ref] for ref in chunk.source_refs if ref in positions)
            for chunk in chunks
            if any(ref in positions for ref in chunk.source_refs)
        ]
        if starts != sorted(starts):
            raise ChunkingError("Chunk source references do not follow document order")

        formula_refs = [
            ref for ref in ordered_refs if ref.startswith("#/texts/") and _is_formula(document, ref)
        ]
        for ref in formula_refs:
            occurrences = sum(ref in chunk.source_refs for chunk in chunks)
            if occurrences != 1:
                raise ChunkingError(
                    f"Formula source {ref} must occur in exactly one chunk; found {occurrences}"
                )

        warnings: List[str] = []
        for chunk in chunks:
            if chunk.token_count > self.options.max_tokens:
                if set(chunk.content_types) <= {"formula"}:
                    warnings.append(
                        f"Atomic formula chunk {chunk.chunk_index} exceeds the token limit"
                    )
                else:
                    raise ChunkingError(
                        f"Chunk {chunk.chunk_index} exceeds {self.options.max_tokens} tokens"
                    )
        return warnings


def _self_ref(item: Any) -> str:
    value = getattr(item, "self_ref", "")
    return str(value or "")


def _provenance(items: Iterable[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for item in items:
        ref = _self_ref(item)
        for prov in list(getattr(item, "prov", []) or []):
            dumped = prov.model_dump(mode="json") if hasattr(prov, "model_dump") else dict(prov)
            if "page_no" in dumped:
                dumped["page_number"] = dumped.pop("page_no")
            dumped["source_ref"] = ref
            result.append(dumped)
    return result


def _unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


def _is_formula(document: Any, ref: str) -> bool:
    try:
        item = document.get_ref(ref).resolve(document)
        return str(getattr(item, "label", "")) == "formula"
    except Exception:
        return False


def _asset_kind(ref: str) -> str | None:
    if ref.startswith("#/pictures/"):
        return "figure"
    if ref.startswith("#/tables/"):
        return "table"
    return None


def _asset_id(document_id: str, ref: str) -> str:
    kind = _asset_kind(ref) or "asset"
    ordinal = int(ref.rsplit("/", 1)[-1]) + 1
    return f"{document_id}:{kind}:{kind}_{ordinal:04d}.png"
