from __future__ import annotations

# Centralized prompt templates for easy iteration.


RISK_AUDITOR_SYSTEM_PROMPT = """You are a strict legal risk critic.
Only use the provided evidence text.
Do not assume facts beyond evidence.
Output concise bullet points in this format:
- [SEVERITY] Risk title | Why it matters | Evidence: <source + section>
If evidence is insufficient, output one bullet:
- [LOW] Insufficient evidence | Ask a clarification question | Evidence: N/A
"""


LOW_CONFIDENCE_CLARIFICATION_PROMPT = (
	"I found low-confidence evidence. Please specify contract name or clause topic "
	"(e.g., liability cap, breach notice, indemnity)."
)


NO_EVIDENCE_CLARIFICATION_PROMPT = (
	"I could not retrieve relevant clauses from the indexed contracts. "
	"Please mention the contract name and clause topic."
)


MISSING_QUERY_PROMPT = "I could not infer your target clause. Please restate your question."


def build_risk_auditor_user_prompt(query: str, evidence_text: str) -> str:
	return (
		f"Question:\n{query}\n\n"
		f"Evidence Clauses:\n{evidence_text}\n\n"
		"Return 3 to 7 bullets only."
	)