"""Prompt contract for structured, citation-grounded research slide decks."""

from __future__ import annotations

from ..models import EvidencePack, GenerationLength, GenerationRequest
from .shared import correction_block, evidence_block, grounding_rules, request_block, revision_block

_DEFAULT_SLIDE_COUNTS = {
    GenerationLength.BRIEF: 3,
    GenerationLength.STANDARD: 5,
    GenerationLength.EXTENDED: 8,
}


def requested_slide_count(request: GenerationRequest) -> int:
    """Return explicit slide count or the product default for the selected duration."""
    return request.slide_count or _DEFAULT_SLIDE_COUNTS[request.length]


def slide_roles(slide_count: int, focus: str) -> list[str]:
    """Build a compact narrative scaffold without forcing paper section headings."""
    if slide_count == 2:
        return ["problem, method, and result", "implications and conclusion"]
    if slide_count == 3:
        return ["problem and objective", "method and main result", "implications and conclusion"]
    roles = [
        "problem and research gap",
        "research objective",
        "method overview",
        "results and comparison",
        "implications, limitations, and conclusion",
    ]
    while len(roles) < slide_count:
        roles.insert(-2, f"supporting technical or empirical evidence for {focus}")
    return roles[:slide_count]


def build_slide_prompt(
    request: GenerationRequest,
    evidence: EvidencePack,
    previous_issues: list[str] | None,
    revision_source: str | None,
) -> str:
    """Build the one deck-level prompt that produces distinct structured slides."""
    count = requested_slide_count(request)
    roles = "\n".join(f"{index}. {role}" for index, role in enumerate(slide_roles(count, request.user_instruction), 1))
    revision_instruction = (
        "Revise the previous deck according to the user instruction. Preserve supported content outside the requested change."
        if revision_source
        else "Create a new deck."
    )
    return f"""You are SARAL's grounded academic presentation generator.

{revision_instruction}
Create exactly {count} slides for {request.audience}. The style is {request.style}.
The user instruction determines emphasis: {request.user_instruction!r}.

{grounding_rules(evidence)}

SLIDE QUALITY
- Give every slide one primary message.
- Use takeaway titles, not labels such as "Methodology" or "Results".
- Use 2-5 concise bullets per slide. Remove generic filler with the so-what test.
- Use named methods, datasets, baselines, and metrics only when supplied by evidence.
- Use equations only when supplied by evidence; preserve LaTeX exactly.
- Give exactly three distinct speaker-note items per slide. Do not repeat bullets verbatim.
- Write a natural spoken script for each slide. Across the deck, match the requested duration.
- Avoid: leverage, synergy, cutting-edge, revolutionary, game-changing, paradigm shift,
  best-in-class, innovative, seamless, holistic, state-of-the-art, novel approach.

AUDIENCE EMPHASIS
- Graduate students: research gap, method, technical setup, equations, experiments, comparisons, limitations.
- Policymakers: problem scale, strongest finding, practical implications, uncertainty, limitations.
- Press/general audience: plain explanation, strongest finding, real-world relevance, caveats.

DECK SCAFFOLD
Use these distinct roles as a narrative guide. Adapt detail to the evidence; never invent missing content.
{roles}

PROVENANCE
- Each slide must include provenance entries for its factual claims.
- A provenance entry contains the claim text and only allowed source chunk IDs.
- Source pages are derived by SARAL from those chunk IDs. Do not supply page numbers yourself.

{request_block(request)}
{revision_block(revision_source)}
{correction_block(previous_issues)}

Return only JSON matching the SlideDeckDraft schema supplied by the API.
{evidence_block(evidence)}"""
