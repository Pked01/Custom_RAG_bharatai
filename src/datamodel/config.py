from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class LLMConfig(BaseModel):
	provider: str
	base_url: str
	api_key_env: str
	model: str
	temperature: float
	top_p: float
	max_tokens: int
	http_referer: Optional[str] = None
	x_title: Optional[str] = None


class EmbeddingConfig(BaseModel):
	provider: str
	base_url: str
	api_key_env: str
	model: str
	http_referer: Optional[str] = None
	x_title: Optional[str] = None


class IngestionConfig(BaseModel):
	raw_dir: str
	persist_dir: str
	collection_name: str
	vector_space: str = "cosine"
	chunk_size: int
	chunk_overlap: int
	reset_collection: bool


class GuardianConfig(BaseModel):
	mode: str = "evaluate_only"
	faithfulness_threshold: float
	max_correction_attempts: int
	focus_drift_penalty: float


class RetrievalConfig(BaseModel):
	semantic_candidate_k: int
	final_top_k: int
	semantic_weight: float
	keyword_weight: float
	section_header_boost: float
	doc_focus_boost: float
	low_confidence_warning_threshold: float
	enable_llm_reranker: bool
	reranker_top_k: int
	reranker_model: str | None = None


class SupervisorConfig(BaseModel):
	"""Configuration for multi-turn query rewriting and context selection."""

	context_overlap_weight: float = 0.85
	context_recency_weight: float = 0.15
	context_top_k: int = 3
	context_max_human_msgs: int = 8


class AppConfig(BaseModel):
	llm: LLMConfig
	embeddings: EmbeddingConfig
	ingestion: IngestionConfig
	guardian: GuardianConfig
	retrieval: RetrievalConfig
	supervisor: SupervisorConfig