from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.researcher import researcher_node
from src.agents.risk_auditor import risk_auditor_node
from src.agents.supervisor import query_rewriter_node
from src.graph.builder import build_graph
from src.ingestion.processor import DocumentProcessor
from src.utils.config import load_config


st.set_page_config(page_title="Custom RAG Workbench", page_icon="🧪", layout="wide")


def _project_root() -> Path:
	return Path(__file__).resolve().parent


def _resolve_path(path_value: str) -> Path:
	path = Path(path_value)
	if path.is_absolute():
		return path
	return _project_root() / path


def _render_ingestion_panel(config_path: Path) -> None:
	st.subheader("Ingestion")
	st.caption("Inspect parsing, semantic chunks, and optional Chroma ingestion independently.")

	try:
		config = load_config(config_path)
	except Exception as exc:
		st.error(f"Unable to load config: {exc}")
		return

	processor = DocumentProcessor(config=config)
	raw_dir = _resolve_path(config.ingestion.raw_dir)
	persist_dir = _resolve_path(config.ingestion.persist_dir)

	col1, col2, col3 = st.columns(3)
	col1.metric("Chunk Size", config.ingestion.chunk_size)
	col2.metric("Chunk Overlap", config.ingestion.chunk_overlap)
	col3.metric("Collection", config.ingestion.collection_name)

	st.write(f"Raw Directory: `{raw_dir}`")
	st.write(f"Chroma Directory: `{persist_dir}`")

	contracts = processor.parse_directory(raw_dir)
	if not contracts:
		st.warning("No contracts found in raw_dir. Add .txt files and refresh.")
		return

	contract_labels = [contract.source_path.name for contract in contracts]
	selected_name = st.selectbox("Select Contract", options=contract_labels)
	selected_contract = next(contract for contract in contracts if contract.source_path.name == selected_name)

	st.markdown(f"### {selected_contract.title}")
	with st.expander("Preamble", expanded=False):
		st.write(selected_contract.preamble or "No preamble detected")

	section_rows = [
		{
			"section_number": section.section_number,
			"section_header": section.section_header,
			"content_chars": len(section.content),
		}
		for section in selected_contract.sections
	]
	st.write("Sections")
	st.dataframe(section_rows, use_container_width=True)

	chunks = processor.create_semantic_chunks(selected_contract)
	st.write(f"Semantic Chunks: {len(chunks)}")
	preview_count = st.slider("Preview chunk count", min_value=1, max_value=max(1, len(chunks)), value=min(3, max(1, len(chunks))))

	for index, chunk in enumerate(chunks[:preview_count]):
		with st.expander(f"Chunk {index}", expanded=False):
			st.json(chunk.metadata)
			st.text(chunk.content)

	if st.button("Run full ingestion to Chroma", type="primary"):
		all_chunks = processor.create_chunks_for_directory(raw_dir)
		try:
			processor.ingest_to_chroma(all_chunks)
			st.success(f"Ingested {len(all_chunks)} chunks into {persist_dir}")
		except Exception as exc:
			st.error(f"Ingestion failed: {exc}")


def _render_placeholder_panel(title: str, reason: str) -> None:
	st.subheader(title)
	st.info(reason)
	st.caption("This section is isolated in the UI so you can develop and test it independently once implementation is added.")


def _state_to_json_safe(state: dict[str, Any]) -> dict[str, Any]:
	json_safe: dict[str, Any] = {}
	for key, value in state.items():
		if key == "messages":
			messages = []
			for message in value:
				if isinstance(message, HumanMessage):
					messages.append({"type": "human", "content": str(message.content)})
				elif isinstance(message, AIMessage):
					messages.append({"type": "ai", "content": str(message.content)})
				else:
					messages.append({"type": type(message).__name__, "content": str(getattr(message, "content", message))})
			json_safe[key] = messages
		else:
			json_safe[key] = value
	return json_safe


def _render_retrieved_clauses(state: dict[str, Any], key_prefix: str = "default") -> None:
	clauses = state.get("retrieved_clauses", [])
	if not clauses:
		st.info("No retrieved clauses in state yet.")
		return

	st.write(f"Retrieved Chunks: {len(clauses)}")
	rows = [
		{
			"file_name": clause.get("file_name", ""),
			"section_number": clause.get("section_number", ""),
			"section_header": clause.get("section_header", ""),
			"chunk_index": clause.get("chunk_index", -1),
			"score": clause.get("score", 0.0),
		}
		for clause in clauses
	]
	st.dataframe(rows, use_container_width=True)

	preview_count = st.slider(
		"Chunk preview count",
		min_value=1,
		max_value=max(1, len(clauses)),
		value=min(3, len(clauses)),
		key=f"{key_prefix}_chunk_preview_count",
	)

	for idx, clause in enumerate(clauses[:preview_count]):
		with st.expander(f"Chunk {idx + 1}"):
			st.caption(f"Source: {clause.get('file_name', '')} | Section: {clause.get('section_number', '')} {clause.get('section_header', '')}")
			st.text(str(clause.get("text", "")))


def _render_risk_report(state: dict[str, Any]) -> None:
	risk_report = state.get("risk_report", [])
	if not risk_report:
		st.info("No risk report generated yet.")
		return

	st.write("Risk Report")
	for line in risk_report:
		st.write(f"- {line}")


def _render_agents_panel() -> None:
	st.subheader("Agents")
	st.caption("Run nodes independently or run full pipeline. Supervisor behavior is implemented through the query rewriter step.")
	st.info(
		"Sequence: 1) Supervisor/Rewriter → 2) Researcher → 3) Risk Auditor. "
		"Use 'Run Full Pipeline' for normal usage."
	)

	if "agent_state" not in st.session_state:
		st.session_state["agent_state"] = {
			"messages": [],
			"retrieved_clauses": [],
			"risk_report": [],
		}

	state: dict[str, Any] = st.session_state["agent_state"]

	user_query = st.text_area("User Query", value=state.get("user_query", ""), height=100)
	current_focus = st.text_input("Current Document Focus (optional)", value=state.get("current_doc_focus", ""))

	col1, col2, col3 = st.columns(3)
	if col1.button("1) Run Supervisor/Rewriter"):
		state["user_query"] = user_query
		if current_focus:
			state["current_doc_focus"] = current_focus
		state["messages"] = [HumanMessage(content=user_query)]
		state.update(query_rewriter_node(state))

	if col2.button("2) Run Researcher"):
		state["user_query"] = user_query
		if current_focus:
			state["current_doc_focus"] = current_focus
		if not state.get("rewritten_query"):
			state.update(query_rewriter_node(state))
		state.update(researcher_node(state))

	if col3.button("3) Run Risk Auditor"):
		state["user_query"] = user_query
		if current_focus:
			state["current_doc_focus"] = current_focus
		state.update(risk_auditor_node(state))

	col4, col5 = st.columns(2)
	if col4.button("Run Full Pipeline", type="primary"):
		state["user_query"] = user_query
		if current_focus:
			state["current_doc_focus"] = current_focus
		state["messages"] = [HumanMessage(content=user_query)]
		state.update(query_rewriter_node(state))
		state.update(researcher_node(state))
		if not state.get("needs_clarification", False):
			state.update(risk_auditor_node(state))

	if col5.button("Reset Agent State"):
		st.session_state["agent_state"] = {
			"messages": [],
			"retrieved_clauses": [],
			"risk_report": [],
		}
		st.rerun()

	st.markdown("### State Snapshot")
	st.json(_state_to_json_safe(st.session_state["agent_state"]))

	st.markdown("### Outputs")
	st.write(f"Rewritten Query: {state.get('rewritten_query', '')}")
	st.write(
		f"Retrieval Confidence: {state.get('retrieval_confidence', 0.0)} "
		f"(warning threshold: {state.get('retrieval_warning_threshold', 0.35)})"
	)
	if state.get("retrieval_warning"):
		st.info(state.get("retrieval_warning"))
	if state.get("needs_clarification"):
		st.warning(state.get("clarifying_question", "Clarification required."))
	_render_retrieved_clauses(state, key_prefix="agents")
	_render_risk_report(state)


def _render_graph_panel() -> None:
	st.subheader("Graph")
	st.caption("Run full LangGraph flow with checkpointed memory by thread_id.")
	st.info("Graph sequence: Supervisor/Rewriter → Researcher → (if evidence exists) Risk Auditor.")

	thread_id = st.text_input("Thread ID", value="demo-thread")
	query = st.text_area("User Query", value="What are the liability risks in the vendor agreements?", height=100)

	if st.button("Run Graph", type="primary"):
		try:
			app = build_graph()
			result = app.invoke(
				{
					"messages": [HumanMessage(content=query)],
					"user_query": query,
				},
				config={"configurable": {"thread_id": thread_id}},
			)
			st.success("Graph executed successfully.")
			st.markdown("### Final State")
			st.json(_state_to_json_safe(result))

			st.markdown("### Outputs")
			st.write(f"Rewritten Query: {result.get('rewritten_query', '')}")
			st.write(
				f"Retrieval Confidence: {result.get('retrieval_confidence', 0.0)} "
				f"(warning threshold: {result.get('retrieval_warning_threshold', 0.35)})"
			)
			if result.get("retrieval_warning"):
				st.info(result.get("retrieval_warning"))
			if result.get("needs_clarification"):
				st.warning(result.get("clarifying_question", "Clarification required."))
			_render_retrieved_clauses(result, key_prefix="graph")
			_render_risk_report(result)
		except Exception as exc:
			st.error(f"Graph run failed: {exc}")


def main() -> None:
	st.title("Custom RAG Workbench")
	st.caption("Visualize and test each subsystem independently.")

	default_config = _project_root() / "src" / "resources" / "config-local.yaml"
	config_path_input = st.sidebar.text_input("Config file", value=str(default_config))
	config_path = Path(config_path_input)

	panel = st.sidebar.radio(
		"Module",
		options=["Ingestion", "Agents", "Graph", "Evaluation"],
	)

	if panel == "Ingestion":
		_render_ingestion_panel(config_path)
	elif panel == "Agents":
		_render_agents_panel()
	elif panel == "Graph":
		_render_graph_panel()
	else:
		_render_placeholder_panel(
			title="Evaluation",
			reason="Evaluator module is currently placeholder. Add evaluation logic to run side-by-side checks here.",
		)


if __name__ == "__main__":
	main()
