"""LangGraph workflow for bounded, citation-grounded document artifacts."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Protocol, TypedDict
from uuid import UUID

from langgraph.graph import END, START, StateGraph
from langsmith import traceable, get_current_run_tree, Client as LangSmithClient
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from .exceptions import GenerationError
from .models import (
    ArtifactCitation,
    ArtifactType,
    ArtifactVisualAsset,
    EvidenceAsset,
    EvidenceChunk,
    EvidencePack,
    GeneratedArtifact,
    GeneratedArtifactDraft,
    GenerationLength,
    GenerationRequest,
    GenerationResponse,
    GenerationSettings,
    GroundingReport,
    RetrievedChunk,
    RetrievalResponse,
    SlideDeckDraft,
)
from .prompts import build_generation_prompt

LOGGER = logging.getLogger(__name__)
_MAX_GENERATION_ATTEMPTS = 2
_CITATION_PATTERN = re.compile(r"\[([^\[\]\s]+)\]")
_WORD_PATTERN = re.compile(r"\b[\w'-]+\b")
_WORD_BUDGETS = {
    GenerationLength.BRIEF: 140,
    GenerationLength.STANDARD: 450,
    GenerationLength.EXTENDED: 1_000,
}
_SLIDE_WORD_BUDGETS = {
    GenerationLength.BRIEF: 220,
    GenerationLength.STANDARD: 650,
    GenerationLength.EXTENDED: 1_400,
}
_SLIDE_QUERY_CANDIDATE_LIMIT = 4
_SLIDE_EVIDENCE_LIMIT = 8
_SLIDE_SECTION_COUNTS = (
    ("background", 1),
    ("objective", 1),
    ("method", 1),
    ("results", 2),
    ("implications", 1),
)


@dataclass(frozen=True)
class _SlideRetrievalRequirement:
    """One deterministic evidence need for a slide-outline request."""

    section: str
    priority: int
    focus: str


@dataclass
class _SlideEvidenceCandidate:
    """A retrieved chunk and the presentation sections it can support."""

    chunk: RetrievedChunk
    sections: set[str] = field(default_factory=set)
    priority: int = 0


class ArtifactGenerator(Protocol):
    """Provider boundary for one structured grounded-artifact draft."""

    def generate(
        self,
        prompt: str,
        max_output_tokens: int,
        response_model: type[BaseModel] = GeneratedArtifactDraft,
    ) -> BaseModel:
        """Generate and parse one draft using the provider's structured-output support."""


class OpenRouterArtifactGenerator:
    """OpenRouter implementation that returns Pydantic-validated artifact drafts."""

    def __init__(self, settings: GenerationSettings) -> None:
        self.settings = settings
        # The SDK retries only transient errors. max_retries=2 permits at most three attempts.
        self.client = OpenAI(
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=2,
        )

    def generate(
        self,
        prompt: str,
        max_output_tokens: int,
        response_model: type[BaseModel] = GeneratedArtifactDraft,
    ) -> BaseModel:
        """Call OpenRouter JSON-schema output and translate provider failures."""
        try:
            completion = self.client.chat.completions.create(
                model=self.settings.model,
                messages=[{"role": "system", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__.lower(),
                        "strict": True,
                        "schema": _openrouter_strict_schema(response_model.model_json_schema()),
                    },
                },
                max_tokens=min(max_output_tokens, self.settings.max_output_tokens),
                temperature=0.2,
                extra_body={"provider": {"require_parameters": True}},
            )
            content = completion.choices[0].message.content
            if not content:
                raise GenerationError("The generation provider returned no structured artifact")
            # --- Phase 1: log token usage to the active LangSmith run ---
            run_tree = get_current_run_tree()
            if run_tree is not None and completion.usage is not None:
                run_tree.add_metadata({
                    "llm_model": self.settings.model,
                    "input_tokens": completion.usage.prompt_tokens,
                    "output_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens,
                    "artifact_schema": response_model.__name__,
                })
            return response_model.model_validate_json(content)
        except GenerationError:
            raise
        except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as exc:
            LOGGER.warning(
                "OpenRouter structured generation request failed",
                extra={
                    "model": self.settings.model,
                    "error_type": type(exc).__name__,
                    "status_code": getattr(exc, "status_code", None),
                },
                exc_info=True,
            )
            raise GenerationError("Grounded generation is temporarily unavailable") from exc
        except (APIStatusError, ValidationError, IndexError) as exc:
            LOGGER.warning(
                "OpenRouter returned an invalid structured generation response",
                extra={
                    "model": self.settings.model,
                    "error_type": type(exc).__name__,
                    "status_code": getattr(exc, "status_code", None),
                },
                exc_info=True,
            )
            raise GenerationError("The generation provider returned an invalid artifact") from exc


def _openrouter_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make every object in a Pydantic schema compatible with strict OpenRouter output."""
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object" or "properties" in value:
                value["additionalProperties"] = False
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    return schema


class GenerationState(TypedDict, total=False):
    """State shared by prompt, generation, grounding, and terminal graph nodes."""

    request: GenerationRequest
    evidence: EvidencePack
    prompt: str
    draft: BaseModel
    grounding: GroundingReport
    attempts: int
    result: GenerationResponse


class GenerationService:
    """Run retrieved evidence through a bounded LangGraph generation workflow."""

    def __init__(self, retriever: Any, generator: ArtifactGenerator) -> None:
        self.retriever = retriever
        self.generator = generator

    def generate(
        self,
        document_id: str,
        owner_id: UUID,
        request: GenerationRequest,
        *,
        retrieval_query: str | None = None,
        revision_source: str | None = None,
    ) -> GenerationResponse:
        """Retrieve document evidence, then return a checked artifact or safe flagged result."""
        started = time.monotonic()
        query = (retrieval_query or request.user_instruction).strip()
        if not query or len(query) > 4_000:
            raise GenerationError("Generation retrieval query is invalid")
        retrieval = (
            self._retrieve_slide_evidence(document_id, owner_id, request, query)
            if request.artifact_type is ArtifactType.SLIDE_OUTLINE
            else self.retriever.retrieve(document_id, owner_id, query, top_k=6)
        )
        if not retrieval.chunks:
            return GenerationResponse(
                document_id=document_id,
                status="unsupported",
                grounding=GroundingReport(
                    passed=False,
                    issues=["The requested artifact is not supported by the retrieved document."],
                ),
                attempts=0,
            )

        evidence = build_evidence_pack(retrieval)
        graph = _build_generation_graph(self.generator, revision_source=revision_source)
        try:
            final_state = graph.invoke({"request": request, "evidence": evidence, "attempts": 0})
            result = final_state["result"]
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError("Grounded generation workflow could not complete") from exc

        LOGGER.info(
            "Grounded generation completed",
            extra={
                "document_id": document_id,
                "owner_id": str(owner_id),
                "status": result.status,
                "attempts": result.attempts,
                "evidence_count": len(evidence.chunks),
                "duration_ms": round((time.monotonic() - started) * 1000),
            },
        )
        return result

    def _retrieve_slide_evidence(
        self,
        document_id: str,
        owner_id: UUID,
        request: GenerationRequest,
        base_query: str,
    ) -> RetrievalResponse:
        """Expand a slide request into bounded section queries and diversify their evidence."""
        candidates: dict[str, _SlideEvidenceCandidate] = {}
        for requirement in _slide_retrieval_requirements(request.audience):
            query = f"{base_query}\nFocus: {requirement.focus}"
            response = self.retriever.retrieve(
                document_id, owner_id, query, top_k=_SLIDE_QUERY_CANDIDATE_LIMIT
            )
            for chunk in response.chunks:
                candidate = candidates.setdefault(chunk.chunk_id, _SlideEvidenceCandidate(chunk=chunk))
                candidate.sections.add(requirement.section)
                candidate.priority = max(candidate.priority, requirement.priority)

        return RetrievalResponse(
            document_id=document_id,
            chunks=_select_slide_evidence(candidates.values()),
        )


def build_evidence_pack(retrieval: RetrievalResponse) -> EvidencePack:
    """Project retrieved chunks into the minimal metadata-preserving prompt contract."""
    chunks = [
        EvidenceChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.contextualized_text or chunk.text,
            page_numbers=chunk.page_numbers,
            heading=chunk.headings[0] if chunk.headings else None,
            provenance=chunk.provenance,
        )
        for chunk in retrieval.chunks
        if (chunk.contextualized_text or chunk.text).strip()
    ]
    if not chunks:
        raise GenerationError("Retrieved chunks did not contain usable document text")
    assets: list[EvidenceAsset] = []
    seen_asset_ids: set[str] = set()
    for chunk in retrieval.chunks:
        for asset_id in chunk.asset_ids:
            if not asset_id or asset_id in seen_asset_ids or len(assets) >= 12:
                continue
            seen_asset_ids.add(asset_id)
            assets.append(
                EvidenceAsset(
                    asset_id=asset_id,
                    source_chunk_id=chunk.chunk_id,
                    page_numbers=chunk.page_numbers,
                    caption=" ".join(chunk.captions).strip() or None,
                )
            )
    return EvidencePack(document_id=retrieval.document_id, chunks=chunks, assets=assets)
def check_grounding(
    draft: BaseModel,
    evidence: EvidencePack,
    length: GenerationLength,
    expected_slide_count: int | None = None,
) -> GroundingReport:
    """Deterministically verify output budget, visible citations, and claim-to-evidence links."""
    if isinstance(draft, SlideDeckDraft):
        return _check_slide_grounding(draft, evidence, length, expected_slide_count)
    if not isinstance(draft, GeneratedArtifactDraft):
        raise GenerationError("Generation returned an unsupported artifact draft")
    allowed = {chunk.chunk_id: chunk for chunk in evidence.chunks}
    issues: list[str] = []
    word_count = len(_WORD_PATTERN.findall(draft.content))
    budget = _WORD_BUDGETS[length]
    if word_count > budget:
        issues.append(f"Artifact has {word_count} words; its {length.value} budget is {budget}.")

    visible_ids = _citation_ids(draft.content)
    unknown_visible = sorted(visible_ids - set(allowed))
    if unknown_visible:
        issues.append("Artifact contains citations not present in the evidence pack: " + ", ".join(unknown_visible))
    if not visible_ids:
        issues.append("Artifact does not include any visible chunk citations.")

    if _uncited_prose_blocks(draft.content):
        issues.append("Artifact has factual prose without a visible chunk citation.")

    claim_ids: set[str] = set()
    for claim in draft.claims:
        claim_id_set = set(claim.citation_ids)
        claim_ids.update(claim_id_set)
        unknown_claim_ids = sorted(claim_id_set - set(allowed))
        if unknown_claim_ids:
            issues.append("Claim cites chunks not present in the evidence pack: " + ", ".join(unknown_claim_ids))
            continue
        if not _claim_has_evidence_overlap(claim.text, [allowed[item].text for item in claim_id_set]):
            issues.append("A claim has insufficient lexical support in its cited evidence.")

    missing_visible_claims = sorted(claim_ids - visible_ids)
    if missing_visible_claims:
        issues.append("Claim citations must also appear in the artifact: " + ", ".join(missing_visible_claims))
    return GroundingReport(
        passed=not issues,
        issues=issues,
        cited_chunk_ids=sorted((visible_ids | claim_ids) & set(allowed)),
    )


def _check_slide_grounding(
    deck: SlideDeckDraft,
    evidence: EvidencePack,
    length: GenerationLength,
    expected_slide_count: int | None,
) -> GroundingReport:
    """Validate slide count, slide provenance, and bounded deck text against evidence."""
    allowed = {chunk.chunk_id: chunk for chunk in evidence.chunks}
    issues: list[str] = []
    if expected_slide_count is not None and len(deck.slides) != expected_slide_count:
        issues.append(
            f"Deck has {len(deck.slides)} slides; the request requires {expected_slide_count}."
        )
    numbers = [slide.slide_number for slide in deck.slides]
    if numbers != list(range(1, len(deck.slides) + 1)):
        issues.append("Slide numbers must start at 1 and be consecutive.")

    deck_text = " ".join(
        [deck.title]
        + [
            " ".join(
                [slide.header_takeaway, *slide.bullets, *slide.speaker_notes, slide.spoken_script]
            )
            for slide in deck.slides
        ]
    )
    word_count = len(_WORD_PATTERN.findall(deck_text))
    budget = _SLIDE_WORD_BUDGETS[length]
    if word_count > budget:
        issues.append(f"Slide deck has {word_count} words; its {length.value} budget is {budget}.")

    cited_ids: set[str] = set()
    for slide in deck.slides:
        slide_ids: set[str] = set()
        for item in slide.provenance:
            item_ids = set(item.citation_ids)
            slide_ids.update(item_ids)
            cited_ids.update(item_ids)
            unknown_ids = sorted(item_ids - set(allowed))
            if unknown_ids:
                issues.append(
                    f"Slide {slide.slide_number} cites chunks not present in the evidence pack: "
                    + ", ".join(unknown_ids)
                )
                continue
            if not _claim_has_evidence_overlap(
                item.claim, [allowed[item_id].text for item_id in item_ids]
            ):
                issues.append(
                    f"Slide {slide.slide_number} has a provenance claim with insufficient evidence overlap."
                )
        if not slide_ids:
            issues.append(f"Slide {slide.slide_number} has no provenance citations.")
    return GroundingReport(
        passed=not issues,
        issues=issues,
        cited_chunk_ids=sorted(cited_ids & set(allowed)),
    )


def _build_generation_graph(generator: ArtifactGenerator, revision_source: str | None = None) -> Any:
    """Compile the fixed workflow: prompt → generate → check → pass/regenerate/flag."""

    @traceable(name="saral.build_prompt", run_type="chain", tags=["saral", "generation", "prompt"])
    def build_prompt_node(state: GenerationState) -> dict[str, Any]:
        previous = state.get("grounding")
        prompt = build_generation_prompt(
            state["request"],
            state["evidence"],
            previous.issues if previous else None,
            revision_source,
        )
        rt = get_current_run_tree()
        if rt is not None:
            rt.add_metadata({
                "artifact_type": state["request"].artifact_type.value,
                "audience": state["request"].audience,
                "length": state["request"].length.value,
                "evidence_chunks": len(state["evidence"].chunks),
                "is_retry": previous is not None,
            })
        return {"prompt": prompt}

    @traceable(name="saral.llm_generate", run_type="llm", tags=["saral", "generation", "openrouter"])
    def generate_node(state: GenerationState) -> dict[str, Any]:
        request = state["request"]
        is_slide_deck = request.artifact_type is ArtifactType.SLIDE_OUTLINE
        # Slides need a generous fixed ceiling because the JSON schema (nested
        # bullets, speaker notes, spoken_script, provenance) is much larger than
        # the grounding word-budget alone. Other artifacts use budget × 3.
        if is_slide_deck:
            max_output_tokens = 6_000
        else:
            budget = _WORD_BUDGETS[request.length]
            max_output_tokens = min(3_600, budget * 3)
        draft = generator.generate(
            state["prompt"],
            max_output_tokens=max_output_tokens,
            response_model=SlideDeckDraft if is_slide_deck else GeneratedArtifactDraft,
        )
        return {"draft": draft, "attempts": state["attempts"] + 1}

    @traceable(name="saral.check_grounding", run_type="tool", tags=["saral", "grounding"])
    def check_node(state: GenerationState) -> dict[str, Any]:
        request = state["request"]
        expected_count = request.slide_count
        if request.artifact_type is ArtifactType.SLIDE_OUTLINE and expected_count is None:
            from .prompts.slides import requested_slide_count

            expected_count = requested_slide_count(request)
        report = check_grounding(state["draft"], state["evidence"], request.length, expected_count)
        # Attach grounding result to the LangSmith run so pass/fail is visible inline.
        rt = get_current_run_tree()
        if rt is not None:
            rt.add_metadata({
                "grounding_passed": report.passed,
                "grounding_issues": report.issues,
                "cited_chunks": len(report.cited_chunk_ids),
                "total_chunks": len(state["evidence"].chunks),
                "citation_coverage_pct": round(
                    100 * len(report.cited_chunk_ids) / max(len(state["evidence"].chunks), 1), 1
                ),
            })
        return {"grounding": report}

    def route_after_check(state: GenerationState) -> Literal["finalize", "build_prompt", "flag"]:
        if state["grounding"].passed:
            return "finalize"
        if state["attempts"] < _MAX_GENERATION_ATTEMPTS:
            return "build_prompt"
        return "flag"

    @traceable(name="saral.finalize", run_type="chain", tags=["saral", "generation"])
    def finalize_node(state: GenerationState) -> dict[str, Any]:
        citations = _citations_from_evidence(state["grounding"].cited_chunk_ids, state["evidence"])
        draft = state["draft"]
        if isinstance(draft, SlideDeckDraft):
            title = draft.title
            content = _slide_deck_markdown(draft)
            deck = draft
        elif isinstance(draft, GeneratedArtifactDraft):
            title = draft.title
            content = draft.content
            deck = None
        else:  # pragma: no cover - provider contract is checked before this node
            raise GenerationError("Generation returned an unsupported artifact draft")
        artifact = GeneratedArtifact(
            artifact_type=state["request"].artifact_type,
            title=title,
            content=content,
            citations=citations,
            deck=deck,
            visual_assets=(
                [
                    ArtifactVisualAsset.model_validate(asset.model_dump(mode="json"))
                    for asset in state["evidence"].assets
                ]
                if state["request"].artifact_type is ArtifactType.SLIDE_OUTLINE
                else []
            ),
        )
        
        # --- Phase 2: Automatic Quality Checks ---
        rt = get_current_run_tree()
        if rt is not None:
            # Import inside node to avoid circular dependency (evaluation depends on generation constants)
            from .evaluation import citation_coverage, claim_overlap_rate, word_budget_adherence
            try:
                ls_client = LangSmithClient()
                ls_client.create_feedback(
                    run_id=rt.id,
                    key="citation_coverage",
                    score=citation_coverage(draft, state["evidence"]),
                )
                ls_client.create_feedback(
                    run_id=rt.id,
                    key="claim_overlap_rate",
                    score=claim_overlap_rate(draft, state["evidence"]),
                )
                budget_result = word_budget_adherence(draft, state["request"].length)
                ls_client.create_feedback(
                    run_id=rt.id,
                    key="word_budget_passed",
                    score=1.0 if budget_result["passed"] else 0.0,
                )
            except Exception as e:
                LOGGER.warning("Failed to log LangSmith evaluation feedback", exc_info=e)

        return {
            "result": GenerationResponse(
                document_id=state["evidence"].document_id,
                status="complete",
                artifact=artifact,
                grounding=state["grounding"],
                attempts=state["attempts"],
            )
        }

    @traceable(name="saral.flag", run_type="chain", tags=["saral", "grounding"])
    def flag_node(state: GenerationState) -> dict[str, Any]:
        return {
            "result": GenerationResponse(
                document_id=state["evidence"].document_id,
                status="flagged",
                grounding=state["grounding"],
                attempts=state["attempts"],
            )
        }

    workflow = StateGraph(GenerationState)
    workflow.add_node("build_prompt", build_prompt_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("check_grounding", check_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("flag", flag_node)
    workflow.add_edge(START, "build_prompt")
    workflow.add_edge("build_prompt", "generate")
    workflow.add_edge("generate", "check_grounding")
    workflow.add_conditional_edges("check_grounding", route_after_check)
    workflow.add_edge("finalize", END)
    workflow.add_edge("flag", END)
    return workflow.compile()


def _citation_ids(content: str) -> set[str]:
    """Return citation IDs rendered in Markdown content."""
    return {match.group(1) for match in _CITATION_PATTERN.finditer(content)}


def _uncited_prose_blocks(content: str) -> list[str]:
    """Return non-heading prose paragraphs and bullets with no visible citation."""
    blocks = [block.strip() for block in re.split(r"\n\s*\n", content) if block.strip()]
    return [
        block
        for block in blocks
        if not all(line.lstrip().startswith("#") for line in block.splitlines())
        and _WORD_PATTERN.search(block)
        and not _citation_ids(block)
    ]


def _claim_has_evidence_overlap(claim: str, evidence_texts: list[str]) -> bool:
    """Reject claim mappings with no meaningful shared vocabulary with the cited chunks."""
    claim_terms = {token.lower() for token in _WORD_PATTERN.findall(claim) if len(token) >= 4}
    evidence_terms = {
        token.lower()
        for evidence in evidence_texts
        for token in _WORD_PATTERN.findall(evidence)
        if len(token) >= 4
    }
    return len(claim_terms & evidence_terms) >= min(2, len(claim_terms)) if claim_terms else False


def _citations_from_evidence(ids: list[str], evidence: EvidencePack) -> list[ArtifactCitation]:
    """Build display citations only from the original retrieved metadata."""
    chunks = {chunk.chunk_id: chunk for chunk in evidence.chunks}
    return [
        ArtifactCitation(
            chunk_id=chunk_id,
            page_numbers=chunks[chunk_id].page_numbers,
            heading=chunks[chunk_id].heading,
        )
        for chunk_id in ids
        if chunk_id in chunks
    ]


def _slide_deck_markdown(deck: SlideDeckDraft) -> str:
    """Create a readable persisted fallback while the UI renders structured slides."""
    blocks = []
    for slide in deck.slides:
        citations = " ".join(
            f"[{citation_id}]" for item in slide.provenance for citation_id in item.citation_ids
        )
        blocks.append(
            f"## Slide {slide.slide_number}: {slide.header_takeaway}\n\n"
            + "\n".join(f"- {bullet}" for bullet in slide.bullets)
            + (f"\n\n{citations}" if citations else "")
        )
    return "\n\n".join(blocks)


def _slide_retrieval_requirements(audience: str) -> tuple[_SlideRetrievalRequirement, ...]:
    """Return cost-bounded query expansion tailored to the requested slide audience."""
    audience_normalized = audience.lower()
    method_focus = (
        "methodology experimental design dataset architecture equations technical implementation"
        if "graduate" in audience_normalized or "technical" in audience_normalized
        else "high-level methodology study design and implementation context"
    )
    implication_focus = (
        "discussion practical implications policy relevance limitations uncertainty conclusion"
        if "policy" in audience_normalized
        else "discussion interpretation limitations uncertainty future work conclusion"
    )
    return (
        _SlideRetrievalRequirement("background", 3, "background motivation problem research gap"),
        _SlideRetrievalRequirement("objective", 3, "research objective question hypothesis stated aim"),
        _SlideRetrievalRequirement("method", 2, method_focus),
        _SlideRetrievalRequirement(
            "results", 5, "results main findings quantitative comparisons baselines figures tables"
        ),
        _SlideRetrievalRequirement("implications", 4, implication_focus),
    )


def _select_slide_evidence(
    candidates: Iterable[_SlideEvidenceCandidate],
    limit: int = _SLIDE_EVIDENCE_LIMIT,
) -> list[RetrievedChunk]:
    """Select unique chunks with section coverage before filling remaining evidence budget."""
    if limit < 1:
        raise GenerationError("Slide evidence limit must be positive")
    candidate_list = list(candidates)
    selected: list[_SlideEvidenceCandidate] = []
    selected_ids: set[str] = set()
    for section, required_count in _SLIDE_SECTION_COUNTS:
        matching = _sorted_slide_candidates(
            candidate
            for candidate in candidate_list
            if section in candidate.sections and candidate.chunk.chunk_id not in selected_ids
        )
        for candidate in matching[:required_count]:
            selected.append(candidate)
            selected_ids.add(candidate.chunk.chunk_id)
    for candidate in _sorted_slide_candidates(candidate_list):
        if len(selected) >= limit:
            break
        if candidate.chunk.chunk_id not in selected_ids:
            selected.append(candidate)
            selected_ids.add(candidate.chunk.chunk_id)
    return [candidate.chunk for candidate in selected[:limit]]


def _sorted_slide_candidates(
    candidates: Iterable[_SlideEvidenceCandidate],
) -> list[_SlideEvidenceCandidate]:
    """Rank candidates deterministically, preferring high-priority visual evidence on ties."""
    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.priority,
            -int(bool(candidate.chunk.asset_ids)),
            -candidate.chunk.rrf_score,
            candidate.chunk.chunk_index,
            candidate.chunk.chunk_id,
        ),
    )
