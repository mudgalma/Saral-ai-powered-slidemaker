"""Docling-native hybrid chunking with explicit grounding invariants."""

from __future__ import annotations

import re
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
        # Build once before the loop: caption-text-ref → figure asset ID.
        # HybridChunker never puts #/pictures/ refs in doc_items, so figures are
        # only reachable via the #/texts/ refs of their captions.
        figure_caption_map = _build_figure_caption_map(document_id, document)
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
            # Direct structural refs cover tables (and hypothetically pictures if ever
            # present in doc_items).  Figure assets come via the caption-ref lookup.
            direct_assets = [_asset_id(document_id, ref) for ref in refs if _asset_kind(ref)]
            figure_assets = [figure_caption_map[ref] for ref in refs if ref in figure_caption_map]
            assets = _unique(direct_assets + figure_assets)
            chunk_captions = list(getattr(meta, "captions", []) or [])
            headings = _clean_headings(
                list(getattr(meta, "headings", []) or []), chunk_captions, labels
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document_id}:{CHUNKER_VERSION}:{index:06d}",
                    document_id=document_id,
                    chunk_index=index,
                    text=raw_chunk.text,
                    contextualized_text=contextualized,
                    headings=headings,
                    captions=chunk_captions,
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


# ── Figure-asset linkage ─────────────────────────────────────────────────────


def _build_figure_caption_map(document_id: str, document: Any) -> Dict[str, str]:
    r"""Return a mapping from each figure's caption text-ref to its asset ID.

    ``HybridChunker`` never places ``#/pictures/N`` refs inside a chunk's
    ``doc_items``; figures are only reachable through the ``#/texts/N`` ref of
    their associated caption item.  By walking the document's ``PictureItem``\s
    once before chunking, we build a lookup that lets ``chunk_document`` inject
    the rendered-figure asset ID into any chunk whose ``source_refs`` include a
    known caption ref.  The ordinal matches ``docling_service._export`` so the
    asset ID is consistent with what is stored in ``figures.json``.
    """
    try:
        from docling_core.types.doc import PictureItem
    except ImportError:  # pragma: no cover - isolated dependency boundary
        return {}
    mapping: Dict[str, str] = {}
    for element, _level in document.iterate_items():
        if not isinstance(element, PictureItem):
            continue
        ref = _self_ref(element)
        if not ref:
            continue
        asset_id = _asset_id(document_id, ref)
        for caption_ref in list(getattr(element, "captions", []) or []):
            cref = str(getattr(caption_ref, "cref", caption_ref) or "")
            if cref:
                mapping[cref] = asset_id
    return mapping


# ── Heading cleanup ───────────────────────────────────────────────────────────

# Matches numbered section headers: "3", "3.2", "3.2.1 Title", etc.
_SECTION_NUMBER_RE = re.compile(r"^\d+(\.[\d]+)*[\s\.]")

# Well-known unnumbered section names in academic papers.
_KNOWN_UNNUMBERED_SECTIONS = frozenset(
    {
        "abstract",
        "introduction",
        "conclusion",
        "conclusions",
        "references",
        "appendix",
        "acknowledgements",
        "acknowledgments",
        "related work",
        "background",
        "discussion",
        "methods",
        "methodology",
    }
)


def _clean_headings(
    headings: List[str], captions: List[str], content_types: List[str]
) -> List[str]:
    """Filter figure-label and axis-text artifacts from the heading list.

    Two sequential filter passes:

    1. **Caption-overlap drop** — any heading string that appears verbatim in
       the chunk's ``captions`` list is a caption that Docling simultaneously
       classified as a heading; drop it.

    2. **Figure-adjacent noise drop** — when ``caption`` or ``picture`` appears
       in ``content_types`` the chunk is known to be figure-adjacent.  Any
       remaining heading that neither matches the section-number prefix pattern
       (e.g. ``3.2.1``) nor is a recognised unnumbered section name is treated
       as a figure sub-label or axis-text artifact and dropped.  This pass is
       skipped for purely narrative chunks so legitimate unnumbered headings
       (e.g. ``Abstract``) are never discarded.
    """
    if not headings:
        return headings
    caption_set = set(captions)
    has_figure_content = any(t in content_types for t in ("caption", "picture"))
    cleaned: List[str] = []
    for h in headings:
        if h in caption_set:
            continue  # literal caption text leaked into heading — drop
        if has_figure_content:
            if (
                not _SECTION_NUMBER_RE.match(h)
                and h.strip().lower() not in _KNOWN_UNNUMBERED_SECTIONS
            ):
                continue  # figure sub-label or axis-text artifact — drop
        cleaned.append(h)
    return cleaned


# ── Table-chunk contract for generation callers ───────────────────────────────


def is_table_chunk(chunk: DocumentChunk) -> bool:
    """Return ``True`` when this chunk is dominated by structural table content.

    Table chunks carry flattened cell text that loses column context when the
    table is split across the token budget.  Callers building narrative
    generation prompts should exclude these chunks and reference the
    ``asset_ids`` image (the full-table crop) directly instead.
    """
    return "table" in chunk.content_types


def narrative_chunks(chunks: Sequence[DocumentChunk]) -> List[DocumentChunk]:
    """Return only the chunks that are safe as narrative generation context.

    Removes table-dominated chunks (prefer their ``asset_ids`` images) while
    preserving text, formula, caption, and mixed-content chunks in source order.
    """
    return [c for c in chunks if not is_table_chunk(c)]
