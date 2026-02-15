from __future__ import annotations

# What this file does:
# - Implements an LLM-as-a-Judge node that validates final answer quality.
# - Scores faithfulness and answer relevancy against retrieved clauses.
# - Sets correction flags and reason for graph-level retry routing.
#
# Why this approach:
# - Keeps QA separate from generation for safer legal outputs.
# - Uses both LLM judgment and deterministic contract-focus checks.
# - Prevents silent hallucinations by forcing explicit correction state.

import json
from functools import lru_cache
from typing import Literal

from src.agents.agent_util import build_evidence_text, get_llm_client
from src.agents.prompts import (
	LEGAL_GUARDIAN_SYSTEM_PROMPT,
	build_legal_guardian_user_prompt,
)
from src.graph.state import AgentState
from src.utils.config import load_guardian_config


@lru_cache(maxsize=1)
def _guardian_config():
	return load_guardian_config()


def _is_reflection_enabled(state: AgentState) -> bool:
	runtime_reflection = state.get("enable_reflection")
	if isinstance(runtime_reflection, bool):
		return runtime_reflection
	mode = (_guardian_config().mode or "evaluate_only").strip().lower()
	return mode == "reflect"


def _extract_json_object(text: str) -> dict:
	text = text.strip()
	if not text:
		return {}

	if text.startswith("```"):
		text = text.strip("`")
		if text.startswith("json"):
			text = text[4:].strip()

	start = text.find("{")
	end = text.rfind("}")
	if start == -1 or end == -1 or end <= start:
		return {}

	candidate = text[start : end + 1]
	try:
		parsed = json.loads(candidate)
		return parsed if isinstance(parsed, dict) else {}
	except Exception:
		return {}


def _score_focus_drift(current_doc_focus: str, retrieved_clauses: list[dict]) -> bool:
	if not current_doc_focus.strip() or not retrieved_clauses:
		return False

	matching = 0
	for clause in retrieved_clauses:
		if str(clause.get("document_title", "")).strip() == current_doc_focus.strip():
			matching += 1
	return matching == 0


def legal_guardian_node(state: AgentState) -> AgentState:
	guardian_config = _guardian_config()
	user_query = (state.get("user_query") or "").strip()
	final_answer = (state.get("final_answer") or "\n".join(state.get("risk_report", []))).strip()
	retrieved_clauses = state.get("retrieved_clauses", [])
	current_doc_focus = (state.get("current_doc_focus") or "").strip()
	correction_attempts = int(state.get("correction_attempts", 0))

	if not final_answer:
		return {
			"faithfulness_score": 0.0,
			"answer_relevancy_score": 0.0,
			"correction_needed": True,
			"correction_reason": "No final answer produced by risk auditor.",
			"correction_attempts": correction_attempts + 1,
		}

	evidence_text = build_evidence_text(retrieved_clauses)
	user_prompt = build_legal_guardian_user_prompt(
		user_query=user_query,
		current_doc_focus=current_doc_focus,
		final_answer=final_answer,
		evidence_text=evidence_text,
	)

	faithfulness_score = 0.0
	answer_relevancy_score = 0.0
	correction_reason = "Guardian could not parse evaluation output."

	try:
		response = get_llm_client(temperature=0.0).invoke(
			[
				("system", LEGAL_GUARDIAN_SYSTEM_PROMPT),
				("human", user_prompt),
			]
		)
		parsed = _extract_json_object(str(response.content))
		faithfulness_score = float(parsed.get("faithfulness_score", 0.0))
		answer_relevancy_score = float(parsed.get("answer_relevancy_score", 0.0))
		correction_reason = str(parsed.get("correction_reason", correction_reason)).strip() or correction_reason
	except Exception as exc:
		correction_reason = f"Guardian evaluation failed: {exc}"

	focus_drift = _score_focus_drift(current_doc_focus=current_doc_focus, retrieved_clauses=retrieved_clauses)
	if focus_drift:
		faithfulness_score = min(faithfulness_score, guardian_config.focus_drift_penalty)
		correction_reason = (
			f"Document focus drift detected. Expected '{current_doc_focus}' but retrieved clauses point to different contract."
		)

	faithfulness_score = max(0.0, min(1.0, faithfulness_score))
	answer_relevancy_score = max(0.0, min(1.0, answer_relevancy_score))

	correction_needed = faithfulness_score < guardian_config.faithfulness_threshold

	return {
		"faithfulness_score": faithfulness_score,
		"answer_relevancy_score": answer_relevancy_score,
		"correction_needed": correction_needed,
		"correction_reason": correction_reason,
		"correction_attempts": correction_attempts + 1 if correction_needed else correction_attempts,
	}


def route_after_legal_guardian(state: AgentState) -> Literal["researcher", "END"]:
	guardian_config = _guardian_config()
	if not _is_reflection_enabled(state):
		return "END"
	if state.get("correction_needed", False):
		attempts = int(state.get("correction_attempts", 0))
		if attempts <= guardian_config.max_correction_attempts:
			return "researcher"
	return "END"
