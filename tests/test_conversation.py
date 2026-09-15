from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from saral_parser.conversation import ConversationService, analyze_intent
from saral_parser.exceptions import GenerationError
from saral_parser.generation import GenerationService
from saral_parser.models import (
    ConversationBranch,
    ConversationMessageRequest,
    GeneratedArtifactDraft,
    GroundedClaim,
    RetrievedChunk,
    RetrievalResponse,
)

OWNER_ID = UUID("11111111-1111-4111-8111-111111111111")
DOCUMENT_ID = "doc_11111111_0123456789abcdef"
THREAD_ID = UUID("22222222-2222-4222-8222-222222222222")


class MemoryConversationPersistence:
    def __init__(self):
        self.messages = []
        self.versions = []

    def ensure_conversation(self, thread_id, document_id, owner_id):
        assert thread_id == THREAD_ID
        assert document_id == DOCUMENT_ID
        assert owner_id == OWNER_ID

    def get_conversation_messages(self, thread_id, document_id, owner_id):
        return list(self.messages)

    def append_conversation_message(self, thread_id, document_id, owner_id, role, content):
        self.messages.append(
            {"id": str(uuid4()), "role": role, "content": content, "created_at": _now()}
        )

    def get_artifact_versions(self, thread_id, document_id, owner_id, limit=20):
        return list(reversed(self.versions[-limit:]))

    def create_artifact_version(
        self, thread_id, document_id, owner_id, artifact, parent_version_id, delta
    ):
        row = {
            "id": str(uuid4()),
            "version_number": len(self.versions) + 1,
            "parent_version_id": str(parent_version_id) if parent_version_id else None,
            "artifact": artifact,
            "delta": delta,
            "created_at": _now(),
        }
        self.versions.append(row)
        return row


class FakeRetriever:
    def retrieve(self, document_id, owner_id, question, top_k):
        assert document_id == DOCUMENT_ID
        assert owner_id == OWNER_ID
        assert top_k == (4 if "Focus:" in question else 6)
        return RetrievalResponse(
            document_id=DOCUMENT_ID,
            chunks=[
                RetrievedChunk(
                    chunk_id="chunk-1",
                    document_id=DOCUMENT_ID,
                    chunk_index=0,
                    text="Evidence supports version one and version two.",
                    contextualized_text="Methods\nEvidence supports version one and version two.",
                    page_numbers=[3],
                    rrf_score=0.1,
                )
            ],
        )


class SequencedGenerator:
    def __init__(self):
        self.calls = 0
        self.prompts = []

    def generate(self, prompt, max_output_tokens, response_model=GeneratedArtifactDraft):
        self.calls += 1
        self.prompts.append(prompt)
        text = f"Evidence supports version {'one' if self.calls == 1 else 'two'}."
        return GeneratedArtifactDraft(
            title=f"Version {self.calls}",
            content=f"{text} [chunk-1]",
            claims=[GroundedClaim(text=text, citation_ids=["chunk-1"])],
        )


def _now():
    return datetime.now(timezone.utc).isoformat()


def _service():
    persistence = MemoryConversationPersistence()
    generator = SequencedGenerator()
    generation = GenerationService(FakeRetriever(), generator)
    return ConversationService(persistence, generation), persistence, generator


def test_conversation_routes_new_generation_then_revision_and_persists_delta():
    service, persistence, generator = _service()
    first = service.respond(
        DOCUMENT_ID,
        OWNER_ID,
        ConversationMessageRequest(thread_id=THREAD_ID, message="Create a script for policymakers"),
    )
    second = service.respond(
        DOCUMENT_ID,
        OWNER_ID,
        ConversationMessageRequest(thread_id=THREAD_ID, message="Make this script shorter"),
    )

    assert first.intent.branch is ConversationBranch.NEW_GENERATION
    assert first.version is not None and first.version.version_number == 1
    assert second.intent.branch is ConversationBranch.REVISION
    assert second.version is not None and second.version.version_number == 2
    assert second.version.parent_version_id == first.version.id
    assert second.version.delta is not None and "previous" in second.version.delta
    assert "<Previous artifact data>" in generator.prompts[1]
    assert [message["role"] for message in persistence.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_conversation_resolves_explicit_version_and_rejects_missing_reference():
    service, _, _ = _service()
    service.respond(
        DOCUMENT_ID,
        OWNER_ID,
        ConversationMessageRequest(thread_id=THREAD_ID, message="Create a script"),
    )
    response = service.respond(
        DOCUMENT_ID,
        OWNER_ID,
        ConversationMessageRequest(thread_id=THREAD_ID, message="Revise #1 in plain English"),
    )

    assert response.version is not None
    assert response.version.parent_version_id is not None
    with pytest.raises(GenerationError, match="referenced artifact version"):
        service.respond(
            DOCUMENT_ID,
            OWNER_ID,
            ConversationMessageRequest(thread_id=THREAD_ID, message="Revise #9"),
        )


def test_intent_analysis_routes_document_question_without_revision_context():
    intent = analyze_intent("What method does the paper use?")

    assert intent.branch is ConversationBranch.QUESTION
    assert intent.artifact_type.value == "answer"
