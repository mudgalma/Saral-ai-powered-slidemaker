"""Prompt contract for grounded Twitter threads and LinkedIn posts."""

from __future__ import annotations

from ..models import EvidencePack, GenerationRequest
from .shared import correction_block, evidence_block, grounding_rules, request_block, revision_block


def build_social_prompt(
    request: GenerationRequest,
    evidence: EvidencePack,
    previous_issues: list[str] | None,
    revision_source: str | None,
) -> str:
    """Build concise, source-only social copy with the existing generic draft schema."""
    format_rules = (
        "Write an 8-10 post thread. Keep each post under 280 characters."
        if request.artifact_type.value == "tweet_thread"
        else "Write one concise professional LinkedIn post under 200 words with a clear takeaway."
    )
    return f"""You create a {request.artifact_type.value} grounded only in supplied document evidence.

{grounding_rules(evidence)}
{format_rules}
- Lead with a supported finding or named method, not generic hype.
- Use clear professional language and avoid the banned marketing phrases in the request context.
- Return title, Markdown content, and claims. Every factual line must end with allowed citations.

{request_block(request)}
{revision_block(revision_source)}
{correction_block(previous_issues)}
{evidence_block(evidence)}"""
