from __future__ import annotations

# What this file does:
# - Keeps orchestration simple and deterministic.
# - Adds a query-rewriter node that resolves ambiguous pronouns using
#   current_doc_focus from prior turns.
# - Provides router helpers used by the graph to decide the next step.
#
# Why this approach:
# - Predictable routing is easier to debug than a fully LLM-driven router.
# - Rewriting before retrieval improves recall for follow-up questions.
# - Thread-level context is preserved by LangGraph checkpointer + thread_id.

import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.state import AgentState


AMBIGUOUS_TERMS = re.compile(r"\b(it|that|this|those|these|the other one|other one)\b", re.IGNORECASE)


def _latest_user_query(state: AgentState) -> str:
	if state.get("user_query"):
		return state["user_query"]

	for message in reversed(state.get("messages", [])):
		if isinstance(message, HumanMessage):
			return str(message.content)
	return ""


def query_rewriter_node(state: AgentState) -> AgentState:
	query = _latest_user_query(state).strip()
	focus = state.get("current_doc_focus", "").strip()

	if not query:
		return {
			"user_query": "",
			"rewritten_query": "",
			"needs_clarification": True,
			"clarifying_question": "Please provide a question to analyze.",
			"messages": [AIMessage(content="Please provide a question to analyze.")],
		}

	rewritten_query = query
	if focus and AMBIGUOUS_TERMS.search(query):
		rewritten_query = AMBIGUOUS_TERMS.sub(focus, query)

	return {
		"user_query": query,
		"rewritten_query": rewritten_query,
	}


def route_after_research(state: AgentState) -> Literal["risk_auditor", "END"]:
	if state.get("needs_clarification", False):
		return "END"
	return "risk_auditor"
