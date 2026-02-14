from __future__ import annotations

# Summary of what this processor does and why:
# - Parses contract text files into title, preamble, and numbered sections.
# - Keeps semantic structure (title + section header + content) in each chunk
#   so retrieval returns legally meaningful context instead of flat text.
# - Stores section-aware metadata (section number/header, source, chunk index)
#   to improve filtering, traceability, and auditability.
# - Uses configured chunking + embeddings + Chroma persistence so ingestion
#   behavior is controlled from config rather than hard-coded in code.

import re
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.datamodel.ingestion import ParsedContract, ParsedSection, SemanticChunk
from src.utils.config import AppConfig, get_api_key, load_config


SECTION_HEADER_PATTERN = re.compile(r"^(?P<number>\d+(?:\.\d+)*)\.\s+(?P<header>.+?)\s*$")


class DocumentProcessor:
	def __init__(self, config: AppConfig | None = None) -> None:
		self.config = config or load_config()
		self.chunk_size = self.config.ingestion.chunk_size
		self.chunk_overlap = self.config.ingestion.chunk_overlap
		self.splitter = RecursiveCharacterTextSplitter(
			chunk_size=self.chunk_size,
			chunk_overlap=self.chunk_overlap,
			separators=["\n\n", "\n", ". ", " ", ""],
		)

	def parse_contract(self, file_path: str | Path) -> ParsedContract:
		path = Path(file_path)
		raw_text = path.read_text(encoding="utf-8")
		lines = [line.rstrip() for line in raw_text.splitlines()]

		title = self._extract_title(lines) or path.stem.replace("_", " ").title()
		preamble, sections = self._extract_sections(lines)

		return ParsedContract(
			source_path=path,
			title=title,
			preamble=preamble,
			sections=sections,
		)

	def parse_directory(self, raw_dir: str | Path) -> list[ParsedContract]:
		root = Path(raw_dir)
		if not root.exists():
			return []

		contracts: list[ParsedContract] = []
		for file_path in sorted(root.glob("*.txt")):
			contracts.append(self.parse_contract(file_path))
		return contracts

	def create_semantic_chunks(self, contract: ParsedContract) -> list[SemanticChunk]:
		chunks: list[SemanticChunk] = []

		if contract.preamble:
			chunks.extend(
				self._chunk_section(
					title=contract.title,
					section_number="0",
					section_header="Preamble",
					content=contract.preamble,
					source_path=contract.source_path,
				)
			)

		for section in contract.sections:
			chunks.extend(
				self._chunk_section(
					title=contract.title,
					section_number=section.section_number,
					section_header=section.section_header,
					content=section.content,
					source_path=contract.source_path,
				)
			)
		return chunks

	def create_chunks_for_directory(self, raw_dir: str | Path) -> list[SemanticChunk]:
		all_chunks: list[SemanticChunk] = []
		for contract in self.parse_directory(raw_dir):
			all_chunks.extend(self.create_semantic_chunks(contract))
		return all_chunks

	def ingest_to_chroma(self, documents: list[SemanticChunk]):
		documents = list(documents)
		if not documents:
			return None

		persist_path = Path(self.config.ingestion.persist_dir)
		if self.config.ingestion.reset_collection and persist_path.exists():
			shutil.rmtree(persist_path)

		vector_documents = [
			Document(page_content=chunk.content, metadata=chunk.metadata)
			for chunk in documents
		]

		embeddings_api_key = get_api_key(self.config.embeddings.api_key_env)
		embeddings = OpenAIEmbeddings(
			model=self.config.embeddings.model,
			api_key=embeddings_api_key,
			base_url=self.config.embeddings.base_url,
		)
		vector_store = Chroma.from_documents(
			documents=vector_documents,
			embedding=embeddings,
			persist_directory=str(persist_path),
			collection_name=self.config.ingestion.collection_name,
		)
		if hasattr(vector_store, "persist"):
			vector_store.persist()
		return vector_store

	def run_ingestion(self):
		chunks = self.create_chunks_for_directory(self.config.ingestion.raw_dir)
		return self.ingest_to_chroma(chunks)

	def _extract_title(self, lines: list[str]) -> str:
		for line in lines:
			cleaned = line.strip()
			if cleaned:
				return cleaned
		return ""

	def _extract_sections(self, lines: list[str]) -> tuple[str, list[ParsedSection]]:
		preamble_lines: list[str] = []
		sections: list[ParsedSection] = []

		active_number = ""
		active_header = ""
		active_content_lines: list[str] = []
		in_section = False
		saw_title = False

		for raw_line in lines:
			line = raw_line.strip()
			if not saw_title and line:
				saw_title = True
				continue

			match = SECTION_HEADER_PATTERN.match(line)
			if match:
				if in_section:
					sections.append(
						ParsedSection(
							section_number=active_number,
							section_header=active_header,
							content=self._normalize_content(active_content_lines),
						)
					)
				active_number = match.group("number")
				active_header = match.group("header")
				active_content_lines = []
				in_section = True
				continue

			if in_section:
				active_content_lines.append(raw_line)
			else:
				preamble_lines.append(raw_line)

		if in_section:
			sections.append(
				ParsedSection(
					section_number=active_number,
					section_header=active_header,
					content=self._normalize_content(active_content_lines),
				)
			)

		filtered_sections = [section for section in sections if section.content]
		preamble = self._normalize_content(preamble_lines)
		return preamble, filtered_sections

	def _normalize_content(self, lines: list[str]) -> str:
		text = "\n".join(line.rstrip() for line in lines)
		text = re.sub(r"\n{3,}", "\n\n", text)
		return text.strip()

	def _chunk_section(
		self,
		title: str,
		section_number: str,
		section_header: str,
		content: str,
		source_path: Path,
	) -> list[SemanticChunk]:
		if not content.strip():
			return []

		section_chunks = self.splitter.split_text(content)
		output: list[SemanticChunk] = []

		for idx, piece in enumerate(section_chunks):
			chunk_text = (
				f"Title: {title}\n"
				f"Section: {section_number}. {section_header}\n"
				f"Content:\n{piece.strip()}"
			)
			metadata = {
				"source": str(source_path),
				"file_name": source_path.name,
				"document_title": title,
				"section_number": section_number,
				"section_header": section_header,
				"chunk_index": idx,
			}
			output.append(SemanticChunk(content=chunk_text, metadata=metadata))

		return output
