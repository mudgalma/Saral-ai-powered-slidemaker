"""Prompt contract for grounded answers, summaries, and scripts."""

from __future__ import annotations

from ..models import EvidencePack, GenerationRequest
from .shared import correction_block, evidence_block, grounding_rules, request_block, revision_block


def build_answer_prompt(
    request: GenerationRequest,
    evidence: EvidencePack,
    previous_issues: list[str] | None,
    revision_source: str | None,
) -> str:
    """Build the generic structured-artifact prompt without slide-only instructions."""
    return f"""You create a {request.artifact_type.value} grounded only in supplied document evidence.

{grounding_rules(evidence)}
- Return a title, Markdown content, and claims. Every factual paragraph or bullet must end with allowed citations.
- Match the requested audience and style. Keep the artifact useful and complete.

{request_block(request)}
{revision_block(revision_source)}
{correction_block(previous_issues)}
{evidence_block(evidence)}"""
