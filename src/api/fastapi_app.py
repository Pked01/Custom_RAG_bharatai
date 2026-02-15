from __future__ import annotations

import uuid
import warnings
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.graph.builder import build_graph


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Relevance scores must be between 0 and 1.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Pydantic serializer warnings.*",
)

app = FastAPI(
    title="Contract Analysis API",
    version="1.0.0",
    description="FastAPI service for multi-agent contract analysis with citations.",
)

THREAD_HISTORY: dict[str, list[dict[str, Any]]] = {}


class ChatQueryRequest(BaseModel):
    """Request payload for asking a contract analysis question."""

    query: str = Field(..., min_length=1, description="Natural-language user question.")
    thread_id: str | None = Field(
        default=None,
        description="Conversation thread id for multi-turn continuity.",
    )
    enable_reflection: bool = Field(
        default=True,
        description="Enable legal-guardian reflection checks.",
    )
    return_chunks: bool = Field(
        default=False,
        description="Include retrieved clause chunks in the response.",
    )
    return_reflection: bool = Field(
        default=False,
        description="Include reflection and correction metadata in the response.",
    )
    verbose: bool = Field(
        default=False,
        description="Include verbose debug metadata in the response.",
    )


class ChatTurn(BaseModel):
    """One persisted conversation turn for a thread."""

    turn_id: str
    created_at: datetime
    query: str
    answer: str
    confidence: str | None = None
    answer_status: str | None = None


class ChatQueryResponse(BaseModel):
    """Response payload for a contract analysis query."""

    thread_id: str
    turn_id: str
    created_at: datetime
    answer: str
    auditor_response: dict[str, Any] = Field(default_factory=dict)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    topic_comparisons: list[dict[str, Any]] = Field(default_factory=list)
    debug: dict[str, Any] | None = None


class ChatHistoryResponse(BaseModel):
    """Conversation history for a single thread."""

    thread_id: str
    turns: list[ChatTurn]


class ResetThreadResponse(BaseModel):
    """Result payload for thread reset operations."""

    thread_id: str
    cleared_turns: int


def _extract_answer(state: dict[str, Any]) -> str:
    """Derive the user-facing answer from graph state in priority order."""

    auditor_response = state.get("auditor_response") or {}
    answer = auditor_response.get("primary_answer") or state.get("final_answer")
    return answer or "No answer was generated."


def _build_debug_payload(state: dict[str, Any], request: ChatQueryRequest) -> dict[str, Any] | None:
    """Build optional debug payload based on request flags."""

    debug: dict[str, Any] = {}

    if request.return_chunks:
        debug["chunks"] = state.get("retrieved_clauses", [])

    if request.return_reflection:
        debug["reflection"] = {
            "faithfulness_score": state.get("faithfulness_score"),
            "answer_relevancy_score": state.get("answer_relevancy_score"),
            "correction_needed": state.get("correction_needed"),
            "correction_reason": state.get("correction_reason"),
            "correction_attempts": state.get("correction_attempts"),
        }

    if request.verbose:
        debug["verbose"] = {
            "rewritten_query": state.get("rewritten_query"),
            "retrieval_confidence": state.get("retrieval_confidence"),
            "retrieval_warning_threshold": state.get("retrieval_warning_threshold"),
            "retrieval_warning": state.get("retrieval_warning"),
            "current_doc_focus": state.get("current_doc_focus"),
            "needs_clarification": state.get("needs_clarification"),
            "clarifying_question": state.get("clarifying_question"),
        }

    return debug or None


@app.get("/health")
def health() -> dict[str, str]:
    """Return service liveness status."""

    return {"status": "ok"}


@app.post("/api/v1/chat/query", response_model=ChatQueryResponse)
def chat_query(request: ChatQueryRequest) -> ChatQueryResponse:
    """Run one analysis turn and return clean answer, citations, and optional debug data."""

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")

    thread_id = request.thread_id or str(uuid.uuid4())
    turn_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    app_graph = build_graph()
    initial_state: dict[str, Any] = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "enable_reflection": request.enable_reflection,
    }

    try:
        state = app_graph.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"graph execution failed: {exc}") from exc

    auditor_response = state.get("auditor_response") or {}
    answer = _extract_answer(state)
    sources = auditor_response.get("citations") or []
    topic_comparisons = auditor_response.get("topic_comparisons") or []
    debug = _build_debug_payload(state, request)

    turn = {
        "turn_id": turn_id,
        "created_at": created_at,
        "query": query,
        "answer": answer,
        "confidence": auditor_response.get("confidence"),
        "answer_status": auditor_response.get("answer_status"),
    }
    THREAD_HISTORY.setdefault(thread_id, []).append(turn)

    return ChatQueryResponse(
        thread_id=thread_id,
        turn_id=turn_id,
        created_at=created_at,
        answer=answer,
        auditor_response=auditor_response,
        sources=sources,
        topic_comparisons=topic_comparisons,
        debug=debug,
    )


@app.get("/api/v1/chat/{thread_id}/history", response_model=ChatHistoryResponse)
def get_chat_history(thread_id: str) -> ChatHistoryResponse:
    """Return all stored turns for a thread id."""

    turns_data = THREAD_HISTORY.get(thread_id, [])
    turns = [ChatTurn(**turn) for turn in turns_data]
    return ChatHistoryResponse(thread_id=thread_id, turns=turns)


@app.delete("/api/v1/chat/{thread_id}", response_model=ResetThreadResponse)
def reset_chat_thread(thread_id: str) -> ResetThreadResponse:
    """Delete all stored turns for the provided thread id."""

    turns_data = THREAD_HISTORY.pop(thread_id, [])
    return ResetThreadResponse(thread_id=thread_id, cleared_turns=len(turns_data))
