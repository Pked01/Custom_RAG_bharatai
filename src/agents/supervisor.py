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
import json
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.agent_util import get_llm_client
from src.agents.prompts import SUPERVISOR_REWRITE_SYSTEM_PROMPT, build_supervisor_rewrite_user_prompt
from src.graph.state import AgentState
from src.utils.config import load_supervisor_config


def _supervisor_config():
	# Cached by the loader's internal YAML read cost being small; keep node simple.
	return load_supervisor_config()


AMBIGUOUS_TERMS = re.compile(r"\b(it|that|this|those|these|the other one|other one)\b", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


def _latest_user_query(state: AgentState) -> str:
	if state.get("user_query"):
		return state["user_query"]

	for message in reversed(state.get("messages", [])):
		if isinstance(message, HumanMessage):
			return str(message.content)
	return ""


def _tokens(text: str) -> set[str]:
	return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text or "")}


def _select_rewrite_context(state: AgentState, latest_query: str) -> str:
	"""Select a small, relevant subset of conversation to help LLM rewriting.

	Keeps this deterministic and cheap: lexical overlap + recency.
	"""
	query_tokens = _tokens(latest_query)
	if not query_tokens:
		query_tokens = set()

	config = _supervisor_config()
	overlap_weight = float(config.context_overlap_weight)
	recency_weight = float(config.context_recency_weight)
	top_k = int(config.context_top_k)
	max_human_msgs = int(config.context_max_human_msgs)

	# Collect human messages (excluding latest) with simple overlap scoring.
	human_messages: list[str] = []
	for message in state.get("messages", []):
		if isinstance(message, HumanMessage):
			content = str(message.content).strip()
			if content:
				human_messages.append(content)

	if not human_messages:
		return ""

	prior = human_messages[:-1] if len(human_messages) > 1 else []
	if not prior:
		return ""

	scored: list[tuple[float, int, str]] = []
	for idx, text in enumerate(prior):
		tokens = _tokens(text)
		overlap = (len(tokens & query_tokens) / max(1, len(query_tokens))) if query_tokens else 0.0
		# Recency bonus: later messages get a slightly higher score.
		recency = (idx + 1) / max(1, len(prior))
		score = overlap_weight * overlap + recency_weight * recency
		scored.append((score, idx, text))

	scored.sort(key=lambda item: item[0], reverse=True)
	selected = [text for _, _, text in scored[: max(1, top_k)]]

	# Always include the most recent prior human message for continuity.
	selected.append(prior[-1])

	# De-duplicate while preserving order.
	seen: set[str] = set()
	unique: list[str] = []
	for text in selected:
		if text in seen:
			continue
		seen.add(text)
		unique.append(text)

	# Keep bounded.
	unique = unique[: max(1, max_human_msgs)]
	return "\n".join(f"- {text}" for text in unique)


def _extract_json_object(text: str) -> dict:
	text = (text or "").strip()
	if not text:
		return {}
	start = text.find("{")
	end = text.rfind("}")
	if start == -1 or end == -1 or end <= start:
		return {}
	try:
		obj = json.loads(text[start : end + 1])
		return obj if isinstance(obj, dict) else {}
	except Exception:
		return {}


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

	# Prefer an LLM rewrite using selected multi-turn context, with a safe fallback.
	context_snippets = _select_rewrite_context(state, latest_query=query)
	user_prompt = build_supervisor_rewrite_user_prompt(query, context_snippets)

	try:
		response = get_llm_client(temperature=0.0).invoke(
			[
				("system", SUPERVISOR_REWRITE_SYSTEM_PROMPT),
				("human", user_prompt),
			]
		)
		parsed = _extract_json_object(str(getattr(response, "content", "")))
		rewritten_query = str(parsed.get("rewritten_query", "")).strip() or query
		needs_clarification = bool(parsed.get("needs_clarification", False))
		clarifying_question = str(parsed.get("clarifying_question", "")).strip()

		if needs_clarification and clarifying_question:
			return {
				"user_query": query,
				"rewritten_query": "",
				"needs_clarification": True,
				"clarifying_question": clarifying_question,
				"messages": [AIMessage(content=clarifying_question)],
			}

		return {
			"user_query": query,
			"rewritten_query": rewritten_query,
		}

	except Exception:
		# Fallback to the prior deterministic heuristic to avoid breaking the pipeline.
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
