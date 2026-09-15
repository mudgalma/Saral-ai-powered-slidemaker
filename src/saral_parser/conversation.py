"""Phase 2 LangGraph workflow for grounded document conversations and versions."""

from __future__ import annotations

import difflib
import logging
import re
import time
from typing import Any, Literal, Optional, TypedDict
from uuid import UUID

from langgraph.graph import END, START, StateGraph
from langsmith import traceable, get_current_run_tree

from .exceptions import GenerationError, PersistenceError
from .generation import GenerationService
from .models import (
    ArtifactType,
    ArtifactVersion,
    ConversationBranch,
    ConversationIntent,
    ConversationMessage,
    ConversationMessageRequest,
    ConversationResponse,
    ConversationState,
    GeneratedArtifact,
    GenerationLength,
    GenerationRequest,
    GenerationResponse,
)

LOGGER = logging.getLogger(__name__)
_VERSION_REFERENCE = re.compile(r"(?:^|\s)#(\d+)\b")
_SLIDE_COUNT_REFERENCE = re.compile(r"\b([2-9]|1\d|20)[ -]?slides?\b", re.IGNORECASE)


class ConversationGraphState(TypedDict, total=False):
    """Raw state shared across one bounded Phase 2 conversation graph execution."""

    document_id: str
    owner_id: UUID
    request: ConversationMessageRequest
    conversation: ConversationState
    intent: ConversationIntent
    target: Optional[ArtifactVersion]
    retrieval_query: str
    generation: GenerationResponse
    response: ConversationResponse


class ConversationService:
    """Route document conversation turns through grounded generation and immutable versions."""

    def __init__(self, persistence: Any, generation: GenerationService) -> None:
        self.persistence = persistence
        self.generation = generation

    def respond(
        self, document_id: str, owner_id: UUID, request: ConversationMessageRequest
    ) -> ConversationResponse:
        """Persist a user turn, route it, and store a completed grounded artifact version."""
        started = time.monotonic()
        self.persistence.ensure_conversation(request.thread_id, document_id, owner_id)
        self.persistence.append_conversation_message(
            request.thread_id, document_id, owner_id, "user", request.message
        )
        graph = _build_conversation_graph(self)
        try:
            final_state = graph.invoke(
                {"document_id": document_id, "owner_id": owner_id, "request": request}
            )
            response = final_state["response"]
        except (GenerationError, PersistenceError):
            raise
        except Exception as exc:
            raise GenerationError("Document conversation workflow could not complete") from exc
        LOGGER.info(
            "Grounded conversation completed",
            extra={
                "document_id": document_id,
                "owner_id": str(owner_id),
                "thread_id": str(request.thread_id),
                "branch": response.intent.branch.value,
                "generation_status": response.generation.status,
                "version": response.version.version_number if response.version else None,
                "duration_ms": round((time.monotonic() - started) * 1000),
            },
        )
        return response

    def load_state(
        self, document_id: str, owner_id: UUID, thread_id: UUID
    ) -> ConversationState:
        """Load bounded messages and immutable versions for one owner-scoped thread."""
        messages = [
            ConversationMessage.model_validate(row)
            for row in self.persistence.get_conversation_messages(thread_id, document_id, owner_id)
        ]
        versions = [
            _artifact_version_from_row(row)
            for row in self.persistence.get_artifact_versions(thread_id, document_id, owner_id)
        ]
        return ConversationState(
            thread_id=thread_id,
            document_id=document_id,
            messages=messages,
            current_artifact=versions[0] if versions else None,
            previous_versions=versions,
        )

    def persist_result(
        self,
        state: ConversationGraphState,
    ) -> ConversationResponse:
        """Save a completed artifact as an immutable version and return its deterministic delta."""
        generation = state["generation"]
        intent = state["intent"]
        conversation = state["conversation"]
        request = state["request"]
        if generation.status != "complete" or generation.artifact is None:
            self.persistence.append_conversation_message(
                request.thread_id,
                state["document_id"],
                state["owner_id"],
                "assistant",
                "The requested information was not supported by the document evidence.",
            )
            return ConversationResponse(
                thread_id=request.thread_id,
                document_id=state["document_id"],
                intent=intent,
                generation=generation,
            )

        parent = state.get("target") or conversation.current_artifact
        previous_content = parent.artifact.content if parent else ""
        delta = _artifact_delta(previous_content, generation.artifact.content) if parent else None
        row = self.persistence.create_artifact_version(
            request.thread_id,
            state["document_id"],
            state["owner_id"],
            generation.artifact.model_dump(mode="json"),
            parent.id if parent else None,
            delta,
        )
        version = _artifact_version_from_row(row)
        self.persistence.append_conversation_message(
            request.thread_id,
            state["document_id"],
            state["owner_id"],
            "assistant",
            generation.artifact.content,
        )
        return ConversationResponse(
            thread_id=request.thread_id,
            document_id=state["document_id"],
            intent=intent,
            generation=generation,
            version=version,
        )


def analyze_intent(
    message: str,
    audience_override: str | None = None,
    length_override: GenerationLength | None = None,
    style_override: str | None = None,
) -> ConversationIntent:
    """Classify a bounded user message without using ungrounded model inference."""
    lowered = message.lower()
    revision = bool(
        _VERSION_REFERENCE.search(message)
        or any(term in lowered for term in ("revise", "rewrite", "shorter", "longer", "this slide", "change"))
    )
    question = not revision and (
        "?" in message
        or lowered.startswith(("what ", "why ", "how ", "where ", "when ", "which ", "explain "))
    )
    branch = (
        ConversationBranch.REVISION
        if revision
        else ConversationBranch.QUESTION
        if question
        else ConversationBranch.NEW_GENERATION
    )
    artifact_type = (
        ArtifactType.SLIDE_OUTLINE
        if any(term in lowered for term in ("slide", "deck", "presentation"))
        else ArtifactType.SCRIPT
        if any(term in lowered for term in ("script", "speech", "talk"))
        else ArtifactType.LINKEDIN_POST
        if "linkedin" in lowered
        else ArtifactType.TWEET_THREAD
        if any(term in lowered for term in ("tweet", "thread", "post"))
        else ArtifactType.ANSWER
        if branch is ConversationBranch.QUESTION
        else ArtifactType.SUMMARY
    )
    audience = (
        "policymakers"
        if "policymaker" in lowered
        else "press"
        if "press" in lowered
        else "graduate students"
        if any(term in lowered for term in ("grad student", "graduate student"))
        else "general audience"
    )
    length = (
        GenerationLength.BRIEF
        if any(term in lowered for term in ("shorter", "brief", "30-second", "30 second"))
        else GenerationLength.EXTENDED
        if any(term in lowered for term in ("longer", "detailed", "5-minute", "5 minute"))
        else GenerationLength.STANDARD
    )
    style = "plain English" if any(term in lowered for term in ("plain", "simple", "non-technical")) else "technical"
    slide_match = _SLIDE_COUNT_REFERENCE.search(message)
    return ConversationIntent(
        branch=branch,
        artifact_type=artifact_type,
        audience=audience_override or audience,
        length=length_override or length,
        style=style_override or style,
        slide_count=int(slide_match.group(1)) if slide_match and artifact_type is ArtifactType.SLIDE_OUTLINE else None,
        is_revision=revision,
    )


def resolve_context(state: ConversationGraphState) -> ArtifactVersion | None:
    """Resolve #N and deictic revision references against persisted artifact versions."""
    match = _VERSION_REFERENCE.search(state["request"].message)
    if match:
        version_number = int(match.group(1))
        for version in state["conversation"].previous_versions:
            if version.version_number == version_number:
                return version
        raise GenerationError("The referenced artifact version does not exist in this conversation")
    if state["intent"].branch is ConversationBranch.REVISION:
        if not state["conversation"].current_artifact:
            raise GenerationError("There is no artifact in this conversation to revise")
        return state["conversation"].current_artifact
    return None


def _build_conversation_graph(service: ConversationService) -> Any:
    """Compile intent/context/router branches into the shared grounded generation boundary."""

    @traceable(name="saral.conversation.load", run_type="tool", tags=["saral", "conversation"])
    def load_node(state: ConversationGraphState) -> dict[str, Any]:
        return {
            "conversation": service.load_state(
                state["document_id"], state["owner_id"], state["request"].thread_id
            )
        }

    @traceable(name="saral.conversation.intent", run_type="chain", tags=["saral", "conversation"])
    def intent_node(state: ConversationGraphState) -> dict[str, Any]:
        request = state["request"]
        intent = analyze_intent(request.message, request.audience, request.length, request.style)
        rt = get_current_run_tree()
        if rt is not None:
            rt.add_metadata({
                "branch": intent.branch.value,
                "artifact_type": intent.artifact_type.value,
                "audience": intent.audience,
                "length": intent.length.value,
                "is_revision": intent.is_revision,
                "slide_count": intent.slide_count,
            })
        return {"intent": intent}

    @traceable(name="saral.conversation.context", run_type="tool", tags=["saral", "conversation"])
    def context_node(state: ConversationGraphState) -> dict[str, Any]:
        return {"target": resolve_context(state)}

    def route(state: ConversationGraphState) -> Literal["new_generation", "revision", "question"]:
        return state["intent"].branch.value

    @traceable(name="saral.conversation.new_generation", run_type="chain", tags=["saral", "conversation"])
    def new_generation_node(state: ConversationGraphState) -> dict[str, Any]:
        return {"retrieval_query": state["request"].message}

    @traceable(name="saral.conversation.revision", run_type="chain", tags=["saral", "conversation"])
    def revision_node(state: ConversationGraphState) -> dict[str, Any]:
        target = state["target"]
        if target is None:
            raise GenerationError("Revision target was not resolved")
        return {"retrieval_query": f"{target.artifact.title} {state['request'].message}"[:4_000]}

    @traceable(name="saral.conversation.question", run_type="chain", tags=["saral", "conversation"])
    def question_node(state: ConversationGraphState) -> dict[str, Any]:
        return {"retrieval_query": state["request"].message}

    @traceable(name="saral.conversation.generate", run_type="chain", tags=["saral", "conversation", "generation"])
    def generate_node(state: ConversationGraphState) -> dict[str, Any]:
        intent = state["intent"]
        request = GenerationRequest(
            artifact_type=intent.artifact_type,
            audience=intent.audience,
            length=intent.length,
            style=intent.style,
            user_instruction=state["request"].message,
            slide_count=intent.slide_count,
        )
        revision_source = state["target"].artifact.content if state.get("target") else None
        return {
            "generation": service.generation.generate(
                state["document_id"],
                state["owner_id"],
                request,
                retrieval_query=state["retrieval_query"],
                revision_source=revision_source,
            )
        }

    @traceable(name="saral.conversation.version", run_type="tool", tags=["saral", "conversation"])
    def version_node(state: ConversationGraphState) -> dict[str, Any]:
        return {"response": service.persist_result(state)}

    workflow = StateGraph(ConversationGraphState)
    workflow.add_node("load_state", load_node)
    workflow.add_node("analyze_intent", intent_node)
    workflow.add_node("resolve_context", context_node)
    workflow.add_node("new_generation", new_generation_node)
    workflow.add_node("revision", revision_node)
    workflow.add_node("question", question_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("version_tracking", version_node)
    workflow.add_edge(START, "load_state")
    workflow.add_edge("load_state", "analyze_intent")
    workflow.add_edge("analyze_intent", "resolve_context")
    workflow.add_conditional_edges("resolve_context", route)
    workflow.add_edge("new_generation", "generate")
    workflow.add_edge("revision", "generate")
    workflow.add_edge("question", "generate")
    workflow.add_edge("generate", "version_tracking")
    workflow.add_edge("version_tracking", END)
    return workflow.compile()


def _artifact_version_from_row(row: dict[str, Any]) -> ArtifactVersion:
    """Convert a persistence row into the validated immutable version contract."""
    return ArtifactVersion(
        id=row["id"],
        version_number=row["version_number"],
        parent_version_id=row.get("parent_version_id"),
        artifact=GeneratedArtifact.model_validate(row["artifact"]),
        delta=row.get("delta"),
        created_at=row["created_at"],
    )


def _artifact_delta(previous: str, current: str) -> str:
    """Return a bounded readable unified diff between two immutable artifact contents."""
    lines = difflib.unified_diff(
        previous.splitlines(), current.splitlines(), fromfile="previous", tofile="current", lineterm=""
    )
    return "\n".join(lines)[:8_000]
