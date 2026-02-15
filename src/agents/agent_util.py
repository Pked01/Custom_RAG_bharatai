from __future__ import annotations

# Shared agent utilities
# - Centralizes repeated dependency/client initialization.
# - Keeps cache behavior in one place for consistent agent performance.

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.utils.config import get_api_key, load_embeddings_config, load_ingestion_config, load_llm_config


def normalize_openai_base_url(base_url: str) -> str:
	normalized = base_url.rstrip("/")
	for suffix in ("/chat/completions", "/embeddings"):
		if normalized.endswith(suffix):
			return normalized[: -len(suffix)]
	return normalized


def openrouter_headers(http_referer: str | None, x_title: str | None) -> dict[str, str] | None:
	headers: dict[str, str] = {}
	if http_referer:
		headers["HTTP-Referer"] = http_referer
	if x_title:
		headers["X-Title"] = x_title
	return headers or None


@lru_cache(maxsize=4)
def get_embeddings_client(model_name: str | None = None) -> OpenAIEmbeddings:
	embeddings_config = load_embeddings_config()
	api_key = get_api_key(embeddings_config.api_key_env)
	headers = openrouter_headers(embeddings_config.http_referer, embeddings_config.x_title)
	return OpenAIEmbeddings(
		model=model_name or embeddings_config.model,
		api_key=api_key,
		base_url=normalize_openai_base_url(embeddings_config.base_url),
		default_headers=headers,
	)


@lru_cache(maxsize=4)
def get_vector_store(embedding_model: str | None = None) -> Chroma:
	ingestion_config = load_ingestion_config()
	return Chroma(
		persist_directory=ingestion_config.persist_dir,
		collection_name=ingestion_config.collection_name,
		embedding_function=get_embeddings_client(embedding_model),
		collection_metadata={"hnsw:space": ingestion_config.vector_space},
	)


@lru_cache(maxsize=4)
def get_llm_client(model_name: str | None = None, temperature: float | None = None) -> ChatOpenAI:
	llm_config = load_llm_config()
	api_key = get_api_key(llm_config.api_key_env)
	headers = openrouter_headers(llm_config.http_referer, llm_config.x_title)
	return ChatOpenAI(
		model=model_name or llm_config.model,
		api_key=api_key,
		base_url=normalize_openai_base_url(llm_config.base_url),
		temperature=llm_config.temperature if temperature is None else temperature,
		top_p=llm_config.top_p,
		max_tokens=llm_config.max_tokens,
		default_headers=headers,
	)


def build_evidence_text(retrieved_clauses: list[dict]) -> str:
	if not retrieved_clauses:
		return "No clauses retrieved."

	rows: list[str] = []
	for idx, clause in enumerate(retrieved_clauses, start=1):
		source = clause.get("file_name") or clause.get("source") or "unknown"
		section_number = clause.get("section_number", "")
		section_header = clause.get("section_header", "")
		text = str(clause.get("text", "")).strip()
		rows.append(
			f"[{idx}] source={source}; section={section_number} {section_header}; score={clause.get('score', 0)}\n{text}"
		)
	return "\n\n".join(rows)