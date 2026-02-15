from __future__ import annotations

# What this file does:
# - Audits retrieved legal clauses with a strict grounded system prompt.
# - Produces schema-validated risk analysis with citations.
# - Enforces hallucination avoidance by requiring explicit "cannot_answer" status.
#
# Why this approach:
# - Pydantic structured output is more reliable than free-form JSON parsing.
# - Explicit answer_status forces model to admit when it doesn't know.
# - Backward compatible with legacy risk_report format.

from typing import Any

from src.agents.agent_util import build_evidence_text, get_llm_client
from src.agents.prompts import RISK_AUDITOR_SYSTEM_PROMPT, build_risk_auditor_user_prompt
from src.datamodel.auditor import AuditorResponse
from src.graph.state import AgentState


def _coerce_to_auditor_response(payload: Any) -> AuditorResponse:
	"""Coerce provider output into validated AuditorResponse."""
	if isinstance(payload, AuditorResponse):
		return payload
	if isinstance(payload, dict):
		return AuditorResponse.model_validate(payload)
	if payload is None:
		raise ValueError("Structured response was empty.")
	if hasattr(payload, "model_dump"):
		return AuditorResponse.model_validate(payload.model_dump())
	return AuditorResponse.model_validate(payload)


def _create_fallback_response(reason: str, suggested_action: str) -> AuditorResponse:
	"""Create a cannot_answer response for failure cases."""
	return AuditorResponse(
		answer_status="cannot_answer",
		primary_answer=reason,
		confidence="low",
		confidence_reason="No valid analysis could be performed.",
		risks=[],
		citations=[],
		unanswered_aspects=[reason],
		suggested_followup=suggested_action,
		needs_human_review=True,
		review_reason="Automated analysis failed; manual review required.",
	)


def risk_auditor_node(state: AgentState) -> AgentState:
	"""Risk Auditor node: analyze retrieved clauses and produce structured risk assessment."""
	query = (state.get("rewritten_query") or state.get("user_query") or "").strip()
	retrieved_clauses = state.get("retrieved_clauses", [])

	# Handle no evidence case
	if not retrieved_clauses:
		fallback = _create_fallback_response(
			reason="No relevant contract clauses were retrieved for this question.",
			suggested_action="Please specify the contract name and clause topic (e.g., 'liability in vendor agreement').",
		)
		return {
			"risk_report": fallback.to_legacy_risk_report(),
			"final_answer": fallback.to_final_answer(),
			"auditor_response": fallback.model_dump(),
		}

	# Build evidence text with numbered citations
	evidence_text = build_evidence_text(retrieved_clauses)
	user_prompt = build_risk_auditor_user_prompt(query, evidence_text)

	try:
		structured_llm = get_llm_client().with_structured_output(AuditorResponse)
		response = structured_llm.invoke(
			[
				("system", RISK_AUDITOR_SYSTEM_PROMPT),
				("human", user_prompt),
			]
		)
		auditor_response = _coerce_to_auditor_response(response)

		return {
			"risk_report": auditor_response.to_legacy_risk_report(),
			"final_answer": auditor_response.to_final_answer(),
			"auditor_response": auditor_response.model_dump(),
		}

	except Exception as exc:
		# Fallback path for models/providers that do not support structured output.
		try:
			fallback_response = get_llm_client().invoke(
				[
					("system", RISK_AUDITOR_SYSTEM_PROMPT),
					("human", user_prompt),
				]
			)
			auditor_response = AuditorResponse.model_validate_json(str(fallback_response.content))
			return {
				"risk_report": auditor_response.to_legacy_risk_report(),
				"final_answer": auditor_response.to_final_answer(),
				"auditor_response": auditor_response.model_dump(),
			}
		except Exception:
			pass

		fallback = _create_fallback_response(
			reason=f"Analysis failed due to system error: {exc}",
			suggested_action="Please try again or contact support if the issue persists.",
		)
		return {
			"risk_report": fallback.to_legacy_risk_report(),
			"final_answer": fallback.to_final_answer(),
			"auditor_response": fallback.model_dump(),
		}
