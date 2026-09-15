from typing import Any

from saral_parser.evaluation import citation_coverage, claim_overlap_rate, word_budget_adherence
from saral_parser.models import (
    EvidenceChunk,
    EvidencePack,
    GeneratedArtifactDraft,
    GenerationLength,
    GroundedClaim,
    SlideDeckDraft,
    SlideDraft,
    SlideProvenance,
)


def _mock_evidence_pack(chunk_ids: list[str]) -> EvidencePack:
    return EvidencePack(
        document_id="doc_123",
        chunks=[
            EvidenceChunk(
                chunk_id=c_id,
                page_numbers=[1],
                text=f"Mock evidence text for {c_id} with some overlap words.",
                heading=None,
            )
            for c_id in chunk_ids
        ],
        assets=[],
    )


def test_citation_coverage():
    evidence = _mock_evidence_pack(["chunk-1", "chunk-2", "chunk-3"])

    # 1. Slide Deck Draft
    draft = SlideDeckDraft(
        title="Test Presentation",
        slides=[
            SlideDraft(
                slide_number=1,
                role="test",
                header_takeaway="test",
                bullets=["test"],
                speaker_notes=["a a a", "b b b", "c c c"],
                spoken_script="test",
                provenance=[
                    SlideProvenance(claim="A valid claim.", citation_ids=["chunk-1"]),
                    SlideProvenance(claim="Another valid claim.", citation_ids=["chunk-2", "chunk-3"]),
                    SlideProvenance(claim="An invalid claim.", citation_ids=["chunk-4"]),
                ],
            ),
            SlideDraft(
                slide_number=2,
                role="test",
                header_takeaway="test",
                bullets=["test"],
                speaker_notes=["a a a", "b b b", "c c c"],
                spoken_script="test",
                provenance=[
                    SlideProvenance(claim="A valid claim.", citation_ids=["chunk-1"]),
                ],
            ),
        ],
    )
    # 3 valid claims out of 4 = 75.0%
    assert abs(citation_coverage(draft, evidence) - (3 / 4)) < 0.01

    # 2. Generated Artifact Draft
    draft2 = GeneratedArtifactDraft(
        title="Test Artifact",
        content="Test content",
        claims=[
            GroundedClaim(text="Valid claim", citation_ids=["chunk-1", "chunk-2"]),
            GroundedClaim(text="Invalid claim", citation_ids=["chunk-99"]),
        ],
    )
    # 1 valid out of 2 = 50.0%
    assert abs(citation_coverage(draft2, evidence) - 0.5) < 0.01


def test_claim_overlap_rate():
    evidence = _mock_evidence_pack(["chunk-1", "chunk-2"])

    draft = GeneratedArtifactDraft(
        title="Test Artifact",
        content="Test content",
        claims=[
            # This should pass because "evidence" and "overlap" are in the mock text
            GroundedClaim(text="There is evidence overlap here.", citation_ids=["chunk-1"]),
            # This should fail because it has no common non-stop words with the evidence
            GroundedClaim(text="completely unrelated assertion", citation_ids=["chunk-2"]),
            # This should fail because the citation is invalid
            GroundedClaim(text="evidence overlap", citation_ids=["chunk-99"]),
        ],
    )

    # 1 passed out of 3 total claims = 33.3%
    assert abs(claim_overlap_rate(draft, evidence) - (1 / 3)) < 0.01


def test_word_budget_adherence():
    draft = GeneratedArtifactDraft(
        title="Short Artifact",
        content=" ".join(["word"] * 50),
        claims=[GroundedClaim(text="Valid claim", citation_ids=["chunk-1"])],
    )
    
    result_brief = word_budget_adherence(draft, GenerationLength.BRIEF)
    assert result_brief["passed"] is True
    assert result_brief["word_count"] == 50  # 50 words

    draft_long = GeneratedArtifactDraft(
        title="Long Artifact",
        content=" ".join(["word"] * 500),
        claims=[GroundedClaim(text="Valid claim", citation_ids=["chunk-1"])],
    )
    result_brief_fail = word_budget_adherence(draft_long, GenerationLength.BRIEF)
    assert result_brief_fail["passed"] is False
    assert result_brief_fail["word_count"] == 500
