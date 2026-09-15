from __future__ import annotations

from pydantic import BaseModel

from .generation import _WORD_BUDGETS, _SLIDE_WORD_BUDGETS, _WORD_PATTERN, _claim_has_evidence_overlap
from .models import EvidencePack, GeneratedArtifactDraft, GenerationLength, SlideDeckDraft


def citation_coverage(draft: BaseModel, evidence: EvidencePack) -> float:
    """Return the fraction of factual claims that have valid retrieved citations."""
    allowed_ids = {chunk.chunk_id for chunk in evidence.chunks}
    claims = []
    
    if isinstance(draft, SlideDeckDraft):
        for slide in draft.slides:
            claims.extend(slide.provenance)
    elif isinstance(draft, GeneratedArtifactDraft):
        claims.extend(draft.claims)
    else:
        return 0.0

    if not claims:
        return 0.0

    valid_claims = 0
    for claim in claims:
        # A claim is validly cited if all its citation_ids are present in the evidence pack.
        # This matches the strict bounding requirement.
        if claim.citation_ids and all(c_id in allowed_ids for c_id in claim.citation_ids):
            valid_claims += 1

    return valid_claims / len(claims)


def claim_overlap_rate(draft: BaseModel, evidence: EvidencePack) -> float:
    """Return the fraction of factual claims that have lexical overlap with their cited evidence."""
    allowed = {chunk.chunk_id: chunk for chunk in evidence.chunks}
    claims = []
    
    if isinstance(draft, SlideDeckDraft):
        for slide in draft.slides:
            for prov in slide.provenance:
                claims.append((prov.claim, prov.citation_ids))
    elif isinstance(draft, GeneratedArtifactDraft):
        for claim in draft.claims:
            claims.append((claim.text, claim.citation_ids))
    else:
        return 0.0

    if not claims:
        return 0.0

    passed_claims = 0
    for text, c_ids in claims:
        if not c_ids or not all(c_id in allowed for c_id in c_ids):
            continue
        evidence_texts = [allowed[c_id].text for c_id in c_ids]
        if _claim_has_evidence_overlap(text, evidence_texts):
            passed_claims += 1

    return passed_claims / len(claims)


def word_budget_adherence(draft: BaseModel, length: GenerationLength) -> dict:
    """Return pass/fail and word count against the defined budget."""
    if isinstance(draft, SlideDeckDraft):
        deck_text = " ".join(
            [draft.title]
            + [
                " ".join(
                    [slide.header_takeaway, *slide.bullets, *slide.speaker_notes, slide.spoken_script]
                )
                for slide in draft.slides
            ]
        )
        word_count = len(_WORD_PATTERN.findall(deck_text))
        budget = _SLIDE_WORD_BUDGETS[length]
    elif isinstance(draft, GeneratedArtifactDraft):
        word_count = len(_WORD_PATTERN.findall(draft.content))
        budget = _WORD_BUDGETS[length]
    else:
        return {"passed": False, "word_count": 0, "budget": 0}

    return {
        "passed": word_count <= budget,
        "word_count": word_count,
        "budget": budget
    }
