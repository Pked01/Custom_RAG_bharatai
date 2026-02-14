from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ParsedSection(BaseModel):
	section_number: str
	section_header: str
	content: str


class ParsedContract(BaseModel):
	source_path: Path
	title: str
	preamble: str
	sections: list[ParsedSection]


class SemanticChunk(BaseModel):
	content: str
	metadata: dict[str, Any]