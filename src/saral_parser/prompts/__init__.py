"""Central prompt routing for bounded, document-grounded artifacts."""

from __future__ import annotations

from .answer import build_answer_prompt
from .slides import build_slide_prompt
from .social import build_social_prompt
from ..models import ArtifactType, EvidencePack, GenerationRequest


def build_generation_prompt(
    request: GenerationRequest,
    evidence: EvidencePack,
    previous_issues: list[str] | None = None,
    revision_source: str | None = None,
) -> str:
    """Route one validated request to its artifact-specific prompt contract."""
    if request.artifact_type is ArtifactType.SLIDE_OUTLINE:
        return build_slide_prompt(request, evidence, previous_issues, revision_source)
    if request.artifact_type in {ArtifactType.TWEET_THREAD, ArtifactType.LINKEDIN_POST}:
        return build_social_prompt(request, evidence, previous_issues, revision_source)
    return build_answer_prompt(request, evidence, previous_issues, revision_source)
