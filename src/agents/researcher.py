from __future__ import annotations

# What this file does:
# - Retrieves top relevant clauses from ChromaDB for the rewritten query.
# - Uses text-embedding-3-large as requested for retrieval embeddings.
# - Returns structured evidence payload for high-precision downstream auditing.
#
# Why this approach:
# - Cached clients avoid repeated initialization overhead in iterative testing.
# - Relevance scores are surfaced so routing can ask clarification when weak.
# - Output is intentionally simple dict-based evidence for transparent tracing.

from collections import Counter

from src.agents.agent_util import get_vector_store
from src.agents.prompts import (
	LOW_CONFIDENCE_CLARIFICATION_PROMPT,
	MISSING_QUERY_PROMPT,
	NO_EVIDENCE_CLARIFICATION_PROMPT,
)
from src.graph.state import AgentState


LOW_CONFIDENCE_WARNING_THRESHOLD = 0.35


def _pick_document_focus(clauses: list[dict]) -> str:
	titles = [str(clause.get("document_title", "")).strip() for clause in clauses]
	titles = [title for title in titles if title]
	if not titles:
		return ""
	return Counter(titles).most_common(1)[0][0]


def researcher_node(state: AgentState) -> AgentState:
	query = (state.get("rewritten_query") or state.get("user_query") or "").strip()
	if not query:
		return {
			"retrieved_clauses": [],
			"retrieval_confidence": 0.0,
			"needs_clarification": True,
			"clarifying_question": MISSING_QUERY_PROMPT,
		}

	results = get_vector_store().similarity_search_with_relevance_scores(query, k=6)
	retrieved_clauses: list[dict] = []

	for document, score in results:
		metadata = document.metadata or {}
		retrieved_clauses.append(
			{
				"text": document.page_content,
				"score": float(score),
				"source": metadata.get("source", ""),
				"file_name": metadata.get("file_name", ""),
				"document_title": metadata.get("document_title", ""),
				"section_number": metadata.get("section_number", ""),
				"section_header": metadata.get("section_header", ""),
				"chunk_index": metadata.get("chunk_index", -1),
			}
		)

	top_scores = [item["score"] for item in retrieved_clauses[:3]]
	confidence = sum(top_scores) / len(top_scores) if top_scores else 0.0
	needs_clarification = not retrieved_clauses

	response: AgentState = {
		"retrieved_clauses": retrieved_clauses,
		"retrieval_confidence": confidence,
		"retrieval_warning_threshold": LOW_CONFIDENCE_WARNING_THRESHOLD,
		"current_doc_focus": _pick_document_focus(retrieved_clauses),
		"needs_clarification": needs_clarification,
	}

	if needs_clarification:
		response["clarifying_question"] = NO_EVIDENCE_CLARIFICATION_PROMPT
	elif confidence < LOW_CONFIDENCE_WARNING_THRESHOLD:
		response["retrieval_warning"] = LOW_CONFIDENCE_CLARIFICATION_PROMPT

	return response
