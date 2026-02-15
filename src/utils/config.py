from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Literal

import yaml

from src.datamodel.config import AppConfig, EmbeddingConfig, GuardianConfig, IngestionConfig, LLMConfig


def _to_bool(value: Any) -> bool:
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		normalized = value.strip().lower()
		if normalized in {"true", "1", "yes", "y", "on"}:
			return True
		if normalized in {"false", "0", "no", "n", "off"}:
			return False
	raise ValueError(f"Invalid boolean value: {value}")


def _read_config_data(config_path: str | Path = "src/resources/config-local.yaml") -> dict[str, Any]:
	path = Path(config_path)
	if not path.exists():
		raise FileNotFoundError(f"Config file not found: {path}")

	with path.open("r", encoding="utf-8") as file:
		data = yaml.safe_load(file) or {}

	if not isinstance(data, dict):
		raise ValueError(f"Config root must be a mapping in: {path}")
	return data


def _get_section(data: dict[str, Any], name: str) -> dict[str, Any]:
	section = data.get(name)
	if not isinstance(section, dict):
		raise ValueError(f"Missing or invalid '{name}' section in config.")
	return section


def _required(section: dict[str, Any], key: str, section_name: str) -> Any:
	value = section.get(key)
	if value is None:
		raise ValueError(f"Missing required key '{section_name}.{key}' in config.")
	return value


def _build_config_from_section(
	section_data: dict[str, Any],
	section_name: str,
	config_class: type,
	converters: dict[str, Callable[[Any], Any]] | None = None,
):
	converters = converters or {}
	model_fields = config_class.model_fields
	required_keys = {name for name, field in model_fields.items() if field.is_required()}
	allowed_keys = set(model_fields.keys())

	missing_keys = [key for key in required_keys if key not in section_data]
	if missing_keys:
		missing_text = ", ".join(f"{section_name}.{key}" for key in sorted(missing_keys))
		raise ValueError(f"Missing required keys in config: {missing_text}")

	extra_keys = [key for key in section_data.keys() if key not in allowed_keys]
	if extra_keys:
		extra_text = ", ".join(f"{section_name}.{key}" for key in sorted(extra_keys))
		raise ValueError(f"Unknown keys in config: {extra_text}")

	normalized_data: dict[str, Any] = {}
	for key in allowed_keys:
		if key not in section_data:
			continue
		value = section_data[key]
		if key in converters:
			value = converters[key](value)
		normalized_data[key] = value

	return config_class(**normalized_data)


def load_llm_config(config_path: str | Path = "src/resources/config-local.yaml") -> LLMConfig:
	data = _read_config_data(config_path)
	llm_data = _get_section(data, "llm")
	return _build_config_from_section(
		section_data=llm_data,
		section_name="llm",
		config_class=LLMConfig,
		converters={
			"provider": str,
			"base_url": str,
			"api_key_env": str,
			"model": str,
			"temperature": float,
			"top_p": float,
			"max_tokens": int,
			"http_referer": lambda value: str(value) if value is not None else None,
			"x_title": lambda value: str(value) if value is not None else None,
		},
	)


def load_embeddings_config(config_path: str | Path = "src/resources/config-local.yaml") -> EmbeddingConfig:
	data = _read_config_data(config_path)
	embeddings_data = _get_section(data, "embeddings")
	return _build_config_from_section(
		section_data=embeddings_data,
		section_name="embeddings",
		config_class=EmbeddingConfig,
		converters={
			"provider": str,
			"base_url": str,
			"api_key_env": str,
			"model": str,
			"http_referer": lambda value: str(value) if value is not None else None,
			"x_title": lambda value: str(value) if value is not None else None,
		},
	)


def load_ingestion_config(config_path: str | Path = "src/resources/config-local.yaml") -> IngestionConfig:
	data = _read_config_data(config_path)
	ingestion_data = _get_section(data, "ingestion")
	return _build_config_from_section(
		section_data=ingestion_data,
		section_name="ingestion",
		config_class=IngestionConfig,
		converters={
			"raw_dir": str,
			"persist_dir": str,
			"collection_name": str,
			"vector_space": lambda value: str(value).strip().lower(),
			"chunk_size": int,
			"chunk_overlap": int,
			"reset_collection": _to_bool,
		},
	)


def load_guardian_config(config_path: str | Path = "src/resources/config-local.yaml") -> GuardianConfig:
	data = _read_config_data(config_path)
	guardian_data = _get_section(data, "guardian")
	return _build_config_from_section(
		section_data=guardian_data,
		section_name="guardian",
		config_class=GuardianConfig,
		converters={
			"enable_reflection": _to_bool,
			"faithfulness_threshold": float,
			"max_correction_attempts": int,
			"focus_drift_penalty": float,
		},
	)


def load_config(
	config_path: str | Path = "src/resources/config-local.yaml",
	section: Literal["all", "llm", "embeddings", "ingestion", "guardian"] = "all",
) -> AppConfig | LLMConfig | EmbeddingConfig | IngestionConfig | GuardianConfig:
	if section == "llm":
		return load_llm_config(config_path)
	if section == "embeddings":
		return load_embeddings_config(config_path)
	if section == "ingestion":
		return load_ingestion_config(config_path)
	if section == "guardian":
		return load_guardian_config(config_path)
	if section != "all":
		raise ValueError("section must be one of: all, llm, embeddings, ingestion, guardian")

	llm = load_llm_config(config_path)
	embeddings = load_embeddings_config(config_path)
	ingestion = load_ingestion_config(config_path)
	guardian = load_guardian_config(config_path)
	return AppConfig(llm=llm, embeddings=embeddings, ingestion=ingestion, guardian=guardian)


def get_api_key(api_key_env: str) -> str:
	value = api_key_env.strip()
	if value.startswith("sk-"):
		return value

	api_key = os.getenv(value, "").strip()
	if not api_key:
		raise ValueError(f"Missing API key. Set environment variable: {value}")
	return api_key
