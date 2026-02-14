from pydantic import BaseModel


class LLMConfig(BaseModel):
	provider: str
	base_url: str
	api_key_env: str
	model: str
	temperature: float
	top_p: float
	max_tokens: int


class EmbeddingConfig(BaseModel):
	provider: str
	base_url: str
	api_key_env: str
	model: str


class IngestionConfig(BaseModel):
	raw_dir: str
	persist_dir: str
	collection_name: str
	chunk_size: int
	chunk_overlap: int
	reset_collection: bool


class AppConfig(BaseModel):
	llm: LLMConfig
	embeddings: EmbeddingConfig
	ingestion: IngestionConfig