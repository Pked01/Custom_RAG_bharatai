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

import json
import re
from collections import Counter
from functools import lru_cache

from src.agents.agent_util import get_llm_client, get_vector_store
from src.agents.prompts import (
	LOW_CONFIDENCE_CLARIFICATION_PROMPT,
	MISSING_QUERY_PROMPT,
	NO_EVIDENCE_CLARIFICATION_PROMPT,
	RETRIEVAL_RERANKER_SYSTEM_PROMPT,
	build_retrieval_reranker_user_prompt,
)
from src.graph.state import AgentState
from src.utils.config import load_retrieval_config


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


@lru_cache(maxsize=1)
def _retrieval_config():
	return load_retrieval_config()


def _tokens(text: str) -> set[str]:
	return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text or "")}


def _keyword_overlap_score(query: str, text: str) -> float:
	query_tokens = _tokens(query)
	if not query_tokens:
		return 0.0
	text_tokens = _tokens(text)
	if not text_tokens:
		return 0.0
	return len(query_tokens & text_tokens) / len(query_tokens)


def _header_match_score(query: str, section_header: str) -> float:
	query_tokens = _tokens(query)
	header_tokens = _tokens(section_header)
	if not query_tokens or not header_tokens:
		return 0.0
	return len(query_tokens & header_tokens) / len(header_tokens)


def _normalize_scores(values: list[float]) -> list[float]:
	if not values:
		return []
	vmin = min(values)
	vmax = max(values)
	if vmin == vmax:
		return [0.5 for _ in values]
	return [(value - vmin) / (vmax - vmin) for value in values]


def _extract_json_object(text: str) -> dict:
	text = text.strip()
	start = text.find("{")
	end = text.rfind("}")
	if start == -1 or end == -1 or end <= start:
		return {}
	try:
		obj = json.loads(text[start : end + 1])
		return obj if isinstance(obj, dict) else {}
	except Exception:
		return {}


def _rerank_with_llm(query: str, candidates: list[dict], model_name: str | None) -> list[dict]:
	if not candidates:
		return candidates

	rows: list[str] = []
	for idx, item in enumerate(candidates, start=1):
		rows.append(
			f"[{idx}] file={item.get('file_name','')}; section={item.get('section_number','')} {item.get('section_header','')}; "
			f"score={item.get('fused_score', 0):.4f}\n{str(item.get('text', ''))[:800]}"
		)

	prompt = build_retrieval_reranker_user_prompt(query, "\n\n".join(rows))
	response = get_llm_client(model_name=model_name, temperature=0.0).invoke(
		[
			("system", RETRIEVAL_RERANKER_SYSTEM_PROMPT),
			("human", prompt),
		]
	)
	parsed = _extract_json_object(str(response.content))
	indices = parsed.get("ranked_indices", [])
	if not isinstance(indices, list):
		return candidates

	ordered: list[dict] = []
	seen: set[int] = set()
	for raw_index in indices:
		if not isinstance(raw_index, int):
			continue
		zero_idx = raw_index - 1
		if 0 <= zero_idx < len(candidates) and zero_idx not in seen:
			ordered.append(candidates[zero_idx])
			seen.add(zero_idx)

	for idx, item in enumerate(candidates):
		if idx not in seen:
			ordered.append(item)

	return ordered


def _pick_document_focus(clauses: list[dict]) -> str:
	titles = [str(clause.get("document_title", "")).strip() for clause in clauses]
	titles = [title for title in titles if title]
	if not titles:
		return ""
	return Counter(titles).most_common(1)[0][0]


def researcher_node(state: AgentState) -> AgentState:
	retrieval_cfg = _retrieval_config()
	query = (state.get("rewritten_query") or state.get("user_query") or "").strip()
	if state.get("correction_needed") and state.get("correction_reason"):
		query = f"{query}\n\nCorrection directive: {state.get('correction_reason', '')}".strip()
	if not query:
		return {
			"retrieved_clauses": [],
			"retrieval_confidence": 0.0,
			"needs_clarification": True,
			"clarifying_question": MISSING_QUERY_PROMPT,
		}

	results = get_vector_store().similarity_search_with_relevance_scores(
		query,
		k=retrieval_cfg.semantic_candidate_k,
	)
	retrieved_candidates: list[dict] = []

	for document, score in results:
		metadata = document.metadata or {}
		retrieved_candidates.append(
			{
				"text": document.page_content,
				"semantic_score": float(score),
				"source": metadata.get("source", ""),
				"file_name": metadata.get("file_name", ""),
				"document_title": metadata.get("document_title", ""),
				"section_number": metadata.get("section_number", ""),
				"section_header": metadata.get("section_header", ""),
				"chunk_index": metadata.get("chunk_index", -1),
			}
		)

	semantic_norm = _normalize_scores([item["semantic_score"] for item in retrieved_candidates])
	current_doc_focus = (state.get("current_doc_focus") or "").strip()

	for idx, item in enumerate(retrieved_candidates):
		keyword_score = _keyword_overlap_score(query, item.get("text", ""))
		header_score = _header_match_score(query, item.get("section_header", ""))
		doc_focus_match = (
			1.0
			if current_doc_focus and str(item.get("document_title", "")).strip() == current_doc_focus
			else 0.0
		)
		fused_score = (
			retrieval_cfg.semantic_weight * semantic_norm[idx]
			+ retrieval_cfg.keyword_weight * keyword_score
			+ retrieval_cfg.section_header_boost * header_score
			+ retrieval_cfg.doc_focus_boost * doc_focus_match
		)
		item["keyword_score"] = keyword_score
		item["header_score"] = header_score
		item["doc_focus_match"] = doc_focus_match
		item["fused_score"] = fused_score

	retrieved_candidates.sort(key=lambda item: item.get("fused_score", 0.0), reverse=True)

	if retrieval_cfg.enable_llm_reranker and retrieved_candidates:
		top_for_rerank = retrieved_candidates[: retrieval_cfg.reranker_top_k]
		tail = retrieved_candidates[retrieval_cfg.reranker_top_k :]
		try:
			top_for_rerank = _rerank_with_llm(query, top_for_rerank, retrieval_cfg.reranker_model)
		except Exception:
			pass
		retrieved_candidates = top_for_rerank + tail

	retrieved_clauses = retrieved_candidates[: retrieval_cfg.final_top_k]
	for item in retrieved_clauses:
		item["score"] = item.get("fused_score", 0.0)

	top_scores = [item["score"] for item in retrieved_clauses[:3]]
	confidence = sum(top_scores) / len(top_scores) if top_scores else 0.0
	needs_clarification = not retrieved_clauses

	response: AgentState = {
		"retrieved_clauses": retrieved_clauses,
		"retrieval_confidence": max(0.0, min(1.0, confidence)),
		"retrieval_warning_threshold": retrieval_cfg.low_confidence_warning_threshold,
		"current_doc_focus": _pick_document_focus(retrieved_clauses),
		"needs_clarification": needs_clarification,
	}

	if needs_clarification:
		response["clarifying_question"] = NO_EVIDENCE_CLARIFICATION_PROMPT
	elif confidence < retrieval_cfg.low_confidence_warning_threshold:
		response["retrieval_warning"] = LOW_CONFIDENCE_CLARIFICATION_PROMPT

	return response
