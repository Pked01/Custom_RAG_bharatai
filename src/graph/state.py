from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
	messages: Annotated[list[AnyMessage], add_messages]
	user_query: str
	rewritten_query: str
	current_doc_focus: str
	retrieved_clauses: list[dict[str, Any]]
	risk_report: list[str]
	retrieval_confidence: float
	retrieval_warning_threshold: float
	retrieval_warning: str
	needs_clarification: bool
	clarifying_question: str
