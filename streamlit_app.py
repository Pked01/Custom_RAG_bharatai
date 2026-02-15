from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.researcher import researcher_node
from src.agents.risk_auditor import risk_auditor_node
from src.agents.supervisor import query_rewriter_node
from src.agents.legal_guardian import legal_guardian_node
from src.graph.builder import build_graph
from src.ingestion.processor import DocumentProcessor
from src.utils.config import load_config, load_guardian_config


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


def _render_mermaid(markup: str, height: int = 320) -> None:
	html = f"""
	<div class=\"mermaid\">\n{markup}\n</div>
	<script type=\"module\">
	  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
	  mermaid.initialize({{ startOnLoad: true }});
	</script>
	"""
	components.html(html, height=height, scrolling=True)


def _render_architecture_panel() -> None:
	st.subheader("Architecture")
	st.caption("Flowcharts from README rendered directly in Streamlit.")

	agent_flow = """
flowchart TD
    A[User Query] --> B[Supervisor / Query Rewriter]
    B --> C[Researcher Retrieval]
    C --> D{Any Evidence Retrieved?}
    D -- No --> E[Clarification Needed]
    D -- Yes --> F[Risk Auditor]
    F --> G[Legal Guardian]
    G --> H{Reflection Enabled and Correction Needed?}
    H -- Yes --> C
    H -- No --> I[Final Output]
"""

	rag_flow = """
flowchart LR
    A[Raw Contracts .txt] --> B[Ingestion Processor]
    B --> C[Semantic Chunks + Metadata]
    C --> D[Embeddings]
    D --> E[(Chroma Vector DB)]
    Q[User Question] --> R[Query Rewriter]
    R --> S[Retriever]
    E --> S
    S --> T[Evidence Chunks]
    T --> U[Risk Auditor LLM]
    U --> V[Legal Guardian LLM-as-Judge]
    V --> W[Final Risk Report]
"""

	ingestion_flow = """
flowchart TD
    A[Load config-local.yaml] --> B[Read raw contracts]
    B --> C[Parse title + preamble + sections]
    C --> D[Create semantic chunks]
    D --> E[Attach legal metadata]
    E --> F[Embed chunks]
    F --> G[Write to Chroma collection]
"""

	st.markdown("### Agent Flow")
	_render_mermaid(agent_flow, height=360)
	with st.expander("Show Mermaid code: Agent Flow"):
		st.code(agent_flow, language="mermaid")

	st.markdown("### Whole RAG Flow")
	_render_mermaid(rag_flow, height=320)
	with st.expander("Show Mermaid code: Whole RAG Flow"):
		st.code(rag_flow, language="mermaid")

	st.markdown("### Ingestion Flow")
	_render_mermaid(ingestion_flow, height=300)
	with st.expander("Show Mermaid code: Ingestion Flow"):
		st.code(ingestion_flow, language="mermaid")


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
	"""Render structured risk report with citations and clickable links."""
	auditor_response = state.get("auditor_response")

	# Handle new structured format
	if auditor_response and isinstance(auditor_response, dict):
		_render_structured_auditor_response(auditor_response, state)
		return

	# Fallback to legacy format
	risk_report = state.get("risk_report", [])
	if not risk_report:
		st.info("No risk report generated yet.")
		return

	st.write("Risk Report (Legacy Format)")
	for line in risk_report:
		st.write(f"- {line}")


def _render_structured_auditor_response(response: dict[str, Any], state: dict[str, Any]) -> None:
	"""Render the new structured auditor response with citations."""
	answer_status = response.get("answer_status", "partial")
	primary_answer = response.get("primary_answer", "")
	confidence = response.get("confidence", "low")
	risks = response.get("risks", [])
	citations = response.get("citations", [])
	unanswered = response.get("unanswered_aspects", [])
	suggested_followup = response.get("suggested_followup")
	needs_human_review = response.get("needs_human_review", False)
	review_reason = response.get("review_reason")

	# Status indicator
	status_colors = {
		"answered": "🟢",
		"partial": "🟡",
		"cannot_answer": "🔴",
		"needs_clarification": "🟠",
	}
	status_icon = status_colors.get(answer_status, "⚪")

	st.markdown("### Analysis Result")

	# Primary answer card
	confidence_badge = {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}.get(confidence, confidence)
	st.markdown(f"**Status:** {status_icon} {answer_status.replace('_', ' ').title()} | **Confidence:** {confidence_badge}")

	if primary_answer:
		st.markdown(f"**Answer:** {primary_answer}")

	if response.get("confidence_reason"):
		st.caption(f"Confidence reason: {response.get('confidence_reason')}")

	# Risks section
	if risks:
		st.markdown("#### Identified Risks")
		for risk in risks:
			severity = risk.get("severity", "low")
			severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")
			title = risk.get("title", "Unnamed Risk")
			description = risk.get("description", "")
			citation_ids = risk.get("citation_ids", [])

			# Build citation links
			citation_refs = ""
			if citation_ids and citations:
				refs = []
				for cid in citation_ids:
					cit = next((c for c in citations if c.get("id") == cid), None)
					if cit:
						refs.append(f"[{cid}]")
				citation_refs = " " + " ".join(refs)

			with st.expander(f"{severity_icon} **{title}** {citation_refs}", expanded=True):
				st.write(description)
				if risk.get("quote_snippet"):
					st.caption(f'"{risk.get("quote_snippet")}"')

	# Citations section with document links
	if citations:
		st.markdown("#### Sources")
		raw_dir = _resolve_path("data/raw")
		for cit in citations:
			cit_id = cit.get("id", "?")
			file_name = cit.get("file_name", "unknown")
			section_num = cit.get("section_number", "")
			section_header = cit.get("section_header", "")
			quote = cit.get("verbatim_quote", "")

			# Build file link
			file_path = raw_dir / file_name
			if file_path.exists():
				file_uri = file_path.as_uri()
				link_text = f"[{file_name}]({file_uri})"
			else:
				link_text = file_name

			with st.expander(f"[{cit_id}] {link_text} § {section_num} — {section_header}"):
				st.markdown(f"**Section:** {section_num} {section_header}")
				st.markdown(f"**Quote:** \"{quote}\"")

	# Unanswered aspects
	if unanswered:
		st.markdown("#### Not Covered")
		for aspect in unanswered:
			st.write(f"- {aspect}")

	# Suggested follow-up
	if suggested_followup:
		st.info(f"**Suggested follow-up:** {suggested_followup}")

	# Human review warning
	if needs_human_review:
		st.warning(f"⚠️ **Human review recommended:** {review_reason or 'Manual verification advised.'}")


def _render_agents_panel() -> None:
	st.subheader("Agents")
	guardian_config = load_guardian_config()
	default_reflection = (guardian_config.mode or "evaluate_only").strip().lower() == "reflect"
	st.caption("Run nodes independently or run full pipeline. Supervisor behavior is implemented through the query rewriter step.")
	st.info(
		"Sequence: 1) Supervisor/Rewriter → 2) Researcher → 3) Risk Auditor → 4) Legal Guardian. "
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
	enable_reflection = st.checkbox(
		"Enable reflection loop (guardian can send flow back to researcher)",
		value=bool(state.get("enable_reflection", default_reflection)),
	)
	state["enable_reflection"] = enable_reflection

	col1, col2, col3, col6 = st.columns(4)
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

	if col6.button("4) Run Legal Guardian"):
		state["user_query"] = user_query
		if current_focus:
			state["current_doc_focus"] = current_focus
		state.update(legal_guardian_node(state))

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
			state.update(legal_guardian_node(state))

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
	st.write(f"Guardian Mode (config): {guardian_config.mode}")
	st.write(f"Reflection Enabled (runtime): {state.get('enable_reflection', default_reflection)}")
	st.write(
		f"Retrieval Confidence: {state.get('retrieval_confidence', 0.0)} "
		f"(warning threshold: {state.get('retrieval_warning_threshold', 0.35)})"
	)
	st.write(
		f"Faithfulness: {state.get('faithfulness_score', 0.0)} | "
		f"Answer Relevancy: {state.get('answer_relevancy_score', 0.0)}"
	)
	if state.get("correction_needed"):
		st.warning(f"Correction needed: {state.get('correction_reason', 'Unspecified reason')}")
	if state.get("retrieval_warning"):
		st.info(state.get("retrieval_warning"))
	if state.get("needs_clarification"):
		st.warning(state.get("clarifying_question", "Clarification required."))
	_render_retrieved_clauses(state, key_prefix="agents")
	_render_risk_report(state)


def _render_graph_panel() -> None:
	"""Render Graph panel as a chatbot UI with scrollable conversation history."""
	st.subheader("Contract Analysis Chat")
	guardian_config = load_guardian_config()
	default_reflection = (guardian_config.mode or "evaluate_only").strip().lower() == "reflect"

	# Initialize chat session state
	if "chat_history" not in st.session_state:
		st.session_state["chat_history"] = []
	if "chat_thread_id" not in st.session_state:
		st.session_state["chat_thread_id"] = "chat-session-1"

	# Sidebar controls
	with st.sidebar:
		st.markdown("### Chat Settings")
		thread_id = st.text_input(
			"Thread ID",
			value=st.session_state["chat_thread_id"],
			key="chat_thread_input",
		)
		st.session_state["chat_thread_id"] = thread_id

		enable_reflection = st.checkbox(
			"Enable reflection loop",
			value=default_reflection,
			key="chat_enable_reflection",
		)

		show_debug = st.checkbox("Show debug info", value=False, key="chat_show_debug")

		if st.button("🗑️ Clear Chat", type="secondary"):
			st.session_state["chat_history"] = []
			st.rerun()

	# Chat info banner
	st.caption(f"Thread: `{thread_id}` | Guardian mode: `{guardian_config.mode}` | Reflection: `{enable_reflection}`")

	# Display chat history (scrollable)
	chat_container = st.container()
	with chat_container:
		for entry in st.session_state["chat_history"]:
			role = entry.get("role", "user")
			content = entry.get("content", "")
			state_snapshot = entry.get("state")

			if role == "user":
				with st.chat_message("user"):
					st.write(content)
			else:
				with st.chat_message("assistant"):
					# Render primary answer
					if state_snapshot:
						auditor_response = state_snapshot.get("auditor_response")
						if auditor_response:
							_render_chat_response(auditor_response, state_snapshot, show_debug)
						else:
							st.write(content)

						# Show retrieval/guardian info
						if show_debug:
							with st.expander("🔍 Debug Info"):
								st.write(f"Rewritten Query: {state_snapshot.get('rewritten_query', '')}")
								st.write(f"Retrieval Confidence: {state_snapshot.get('retrieval_confidence', 0.0):.2f}")
								st.write(f"Faithfulness: {state_snapshot.get('faithfulness_score', 0.0):.2f}")
								st.write(f"Answer Relevancy: {state_snapshot.get('answer_relevancy_score', 0.0):.2f}")
								if state_snapshot.get("correction_needed"):
									st.warning(f"Correction needed: {state_snapshot.get('correction_reason')}")
					else:
						st.write(content)

	# Chat input
	user_input = st.chat_input("Ask about your contracts...")

	if user_input:
		# Add user message to history
		st.session_state["chat_history"].append({"role": "user", "content": user_input})

		# Run graph
		try:
			with st.spinner("Analyzing contracts..."):
				app = build_graph()
				result = app.invoke(
					{
						"messages": [HumanMessage(content=user_input)],
						"user_query": user_input,
						"enable_reflection": enable_reflection,
					},
					config={"configurable": {"thread_id": thread_id}},
				)

			# Extract assistant response
			final_answer = result.get("final_answer", "I couldn't generate a response.")

			# Add assistant response to history
			st.session_state["chat_history"].append({
				"role": "assistant",
				"content": final_answer,
				"state": result,
			})

		except Exception as exc:
			error_msg = f"Error: {exc}"
			st.session_state["chat_history"].append({
				"role": "assistant",
				"content": error_msg,
				"state": None,
			})

		st.rerun()


def _render_chat_response(auditor_response: dict[str, Any], state: dict[str, Any], show_debug: bool) -> None:
	"""Render a clean, user-friendly chat response with linked inline citations."""
	answer_status = auditor_response.get("answer_status", "partial")
	primary_answer = auditor_response.get("primary_answer", "")
	confidence = auditor_response.get("confidence", "low")
	is_comparison_query = auditor_response.get("is_comparison_query", False)
	topic_comparisons = auditor_response.get("topic_comparisons", [])
	risks = auditor_response.get("risks", [])
	citations = auditor_response.get("citations", [])
	suggested_followup = auditor_response.get("suggested_followup")
	needs_human_review = auditor_response.get("needs_human_review", False)
	raw_dir = _resolve_path("data/raw")

	# ─────────────────────────────────────────────────────────────
	# Build citation lookup: id -> {file_name, section, quote, link}
	# ─────────────────────────────────────────────────────────────
	citation_map: dict[int, dict[str, Any]] = {}
	for cit in citations:
		cit_id = cit.get("id")
		if cit_id is None:
			continue
		file_name = cit.get("file_name", "unknown")
		section = f"§{cit.get('section_number', '')} {cit.get('section_header', '')}".strip()
		quote = cit.get("verbatim_quote", "")

		# Build file link
		file_path = raw_dir / file_name
		if file_path.exists():
			file_uri = file_path.as_uri()
			link = f"[{file_name}]({file_uri})"
		else:
			link = file_name

		citation_map[cit_id] = {
			"file_name": file_name,
			"section": section,
			"quote": quote,
			"link": link,
		}

	def _linkify_citations(text: str) -> str:
		"""Replace [N] with superscript citations."""
		def replace_cite(match):
			cite_num = int(match.group(1))
			if cite_num in citation_map:
				return f"<sup>[{cite_num}]</sup>"
			return match.group(0)
		return re.sub(r"\[(\d+)\]", replace_cite, text)

	# ─────────────────────────────────────────────────────────────
	# 1. PRIMARY ANSWER — With inline citation superscripts
	# ─────────────────────────────────────────────────────────────
	if answer_status == "cannot_answer":
		st.warning(primary_answer)
	elif answer_status == "needs_clarification":
		st.info(primary_answer)
	else:
		linked_answer = _linkify_citations(primary_answer)
		st.markdown(linked_answer, unsafe_allow_html=True)

	# ─────────────────────────────────────────────────────────────
	# 2. COMPARISON TABLE (for cross-document queries)
	# ─────────────────────────────────────────────────────────────
	if is_comparison_query and topic_comparisons:
		st.markdown("---")
		for comp in topic_comparisons:
			topic = comp.get("topic", "Unknown Topic")
			positions = comp.get("positions", [])
			has_conflict = comp.get("has_conflict", False)
			conflict_desc = comp.get("conflict_description", "")

			table_data: list[dict[str, str]] = []
			for pos in positions:
				file_name = pos.get("file_name", "unknown")
				position_text = pos.get("position", "")
				table_data.append({"Document": file_name, "States": position_text})

			conflict_indicator = "⚠️ Conflict" if has_conflict else "✓ Aligned"
			st.markdown(f"**{topic}** — {conflict_indicator}")

			if table_data:
				df = pd.DataFrame(table_data)
				st.dataframe(df, use_container_width=True, hide_index=True)

			if has_conflict and conflict_desc:
				st.caption(f"↳ {conflict_desc}")

	# ─────────────────────────────────────────────────────────────
	# 3. KEY FINDINGS (for risk queries)
	# ─────────────────────────────────────────────────────────────
	if risks and not is_comparison_query:
		st.markdown("---")
		top_risk = risks[0] if risks else None
		if top_risk:
			title = top_risk.get("title", "")
			desc = top_risk.get("description", "")
			st.markdown(f"**Key Finding:** {title}")
			linked_desc = _linkify_citations(desc)
			st.markdown(linked_desc, unsafe_allow_html=True)

		if len(risks) > 1:
			with st.expander(f"See {len(risks) - 1} more finding(s)"):
				for risk in risks[1:]:
					desc = risk.get("description", "")
					linked_desc = _linkify_citations(desc)
					st.markdown(f"• **{risk.get('title', '')}**: {linked_desc}", unsafe_allow_html=True)

	# ─────────────────────────────────────────────────────────────
	# 4. SOURCES — Linked to inline [N] references
	# ─────────────────────────────────────────────────────────────
	if citation_map:
		st.markdown("---")
		st.markdown("**Sources:**")
		for cit_id in sorted(citation_map.keys()):
			cit = citation_map[cit_id]
			st.markdown(f"**[{cit_id}]** {cit['link']} {cit['section']}")
			if show_debug and cit.get("quote"):
				quote = cit["quote"]
				st.caption(f'> "{quote[:200]}{"..." if len(quote) > 200 else ""}"')

	# ─────────────────────────────────────────────────────────────
	# 5. FOLLOW-UP & REVIEW FLAGS
	# ─────────────────────────────────────────────────────────────
	if suggested_followup:
		st.info(f"💡 **Suggested next step:** {suggested_followup}")

	if needs_human_review:
		st.warning(f"⚠️ **Review recommended:** {auditor_response.get('review_reason', 'Manual verification advised')}")

	if show_debug:
		conf_icons = {"high": "🟢", "medium": "🟡", "low": "🔴"}
		st.caption(f"Status: {answer_status} | Confidence: {conf_icons.get(confidence, '')} {confidence}")


def main() -> None:
	st.title("Custom RAG Workbench")
	st.caption("Visualize and test each subsystem independently.")

	default_config = _project_root() / "src" / "resources" / "config-local.yaml"
	config_path_input = st.sidebar.text_input("Config file", value=str(default_config))
	config_path = Path(config_path_input)

	panel = st.sidebar.radio(
		"Module",
		options=["Ingestion", "Agents", "Graph", "Architecture", "Evaluation"],
	)

	if panel == "Ingestion":
		_render_ingestion_panel(config_path)
	elif panel == "Agents":
		_render_agents_panel()
	elif panel == "Graph":
		_render_graph_panel()
	elif panel == "Architecture":
		_render_architecture_panel()
	else:
		_render_placeholder_panel(
			title="Evaluation",
			reason="Evaluator module is currently placeholder. Add evaluation logic to run side-by-side checks here.",
		)


if __name__ == "__main__":
	main()
