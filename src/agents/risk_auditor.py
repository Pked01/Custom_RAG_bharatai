from __future__ import annotations

# What this file does:
# - Audits retrieved legal clauses with a strict "Critic" system prompt.
# - Produces a concise, citation-oriented risk report for each query.
#
# Why this approach:
# - Auditor consumes only retrieved evidence for precision and traceability.
# - Cached LLM client keeps multi-turn testing responsive.
# - Includes deterministic fallback so the node never fails silently.

from src.agents.agent_util import build_evidence_text, get_llm_client
from src.agents.prompts import RISK_AUDITOR_SYSTEM_PROMPT, build_risk_auditor_user_prompt
from src.graph.state import AgentState


def risk_auditor_node(state: AgentState) -> AgentState:
	query = (state.get("rewritten_query") or state.get("user_query") or "").strip()
	retrieved_clauses = state.get("retrieved_clauses", [])

	if not retrieved_clauses:
		return {
			"risk_report": [
				"[LOW] Insufficient evidence | Please clarify contract and clause focus | Evidence: N/A"
			]
		}

	evidence_text = build_evidence_text(retrieved_clauses)
	user_prompt = build_risk_auditor_user_prompt(query, evidence_text)

	try:
		response = get_llm_client().invoke(
			[
				("system", RISK_AUDITOR_SYSTEM_PROMPT),
				("human", user_prompt),
			]
		)
		raw_text = str(response.content).strip()
		risk_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
		if not risk_lines:
			risk_lines = [
				"[LOW] Insufficient structured output | Re-run with clearer question | Evidence: N/A"
			]
		return {"risk_report": risk_lines}
	except Exception as exc:
		return {
			"risk_report": [
				f"[LOW] Auditor fallback activated | LLM call failed: {exc} | Evidence: Retrieved clauses only"
			]
		}
