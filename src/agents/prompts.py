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


LEGAL_GUARDIAN_SYSTEM_PROMPT = """You are a legal QA judge validating answer quality.
Evaluate only these two metrics:
1) faithfulness_score in [0,1]: Are claims grounded in retrieved clauses?
2) answer_relevancy_score in [0,1]: Does the answer address the user's question?

Use current_doc_focus to check contract drift:
- If answer appears to rely on a different contract than current_doc_focus, reduce faithfulness.

Return strict JSON with keys:
{
  "faithfulness_score": float,
  "answer_relevancy_score": float,
  "correction_reason": string
}
No markdown. No extra keys.
"""


def build_legal_guardian_user_prompt(
	user_query: str,
	current_doc_focus: str,
	final_answer: str,
	evidence_text: str,
) -> str:
	return (
		f"User Query:\n{user_query}\n\n"
		f"Current Document Focus:\n{current_doc_focus or 'N/A'}\n\n"
		f"Final Answer To Validate:\n{final_answer or 'N/A'}\n\n"
		f"Retrieved Clauses:\n{evidence_text or 'N/A'}"
	)