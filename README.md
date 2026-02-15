# Custom RAG Bharatai

A contract-focused Multi-Agent RAG system for legal document ingestion, retrieval, and risk-oriented analysis.

## 1) Process Overview

The project is designed around two major pipelines:

- **Ingestion pipeline**: parse raw contracts, preserve legal structure, chunk semantically, embed, and persist into Chroma.
- **Query pipeline (Agent/Graph)**: rewrite ambiguous follow-up queries, retrieve relevant clauses, generate a risk-oriented answer, and validate output quality through a Legal Guardian node.

### Key Design Principles

- Preserve legal structure (`document_title`, `section_number`, `section_header`) during ingestion.
- Keep retrieval and auditing separate for transparency.
- Support multi-turn context with LangGraph checkpointing (`thread_id`).
- Provide interactive inspection through Streamlit.

---

## 2) Agents and Responsibilities

### Supervisor / Query Rewriter

- Reads current conversation state and user query.
- Rewrites ambiguous follow-up questions into clearer standalone queries.
- Helps maintain continuity with prior `current_doc_focus`.

### Researcher

- Retrieves top candidate chunks from Chroma using hybrid retrieval (semantic similarity + lexical/header/doc-focus signals).
- Returns structured evidence objects with metadata and scores.
- Supports optional LLM reranking over top candidates.
- Sets retrieval confidence and warning/clarification signals.

### Risk Auditor

- Consumes retrieved evidence only.
- Produces structured risk-style findings with severity and citations.
- Uses a strict critic-style system prompt.

### Legal Guardian

- Evaluates the Risk Auditor output for faithfulness and relevancy against retrieved evidence.
- Produces guardian scores and pass/fail control flags.
- Can trigger correction loops when `guardian.mode=reflect` and thresholds are not met.

---

## 3) Agent Flow Chart

![Mermaid flowchart illustrating the agent interactions](docs/images/agent_flow_chart.png)

### Mermaid Source (Agent Flow)

```mermaid
flowchart TD
   U[User Query] --> S[Supervisor / Query Rewriter]
   S --> R[Researcher]
   R -->|Retrieve evidence + confidence| A[Risk Auditor]
   A -->|Draft risk report + final answer| G[Legal Guardian]

   G --> D{Guardian mode}
   D -->|off| OUT[Return auditor output]
   D -->|evaluate_only| E{Pass thresholds?}
   D -->|reflect| F{Pass thresholds?}

   E -->|yes| OUT
   E -->|no| OUTW[Return output + guardian warning]

   F -->|yes| OUT
   F -->|no| C[Set correction flags]
   C --> R2[Re-run Researcher with guidance]
   R2 --> A2[Re-run Risk Auditor]
   A2 --> G2[Re-evaluate in Legal Guardian]
   G2 --> L{Reflection limit reached?}
   L -->|no, still failing| C
   L -->|yes| OUTW
   G2 -->|pass| OUT
```


---

## 4) Whole RAG Flow Chart

![Mermaid flowchart illustrating the whole RAG flow](docs/images/whole_rag_chart.png)

### Mermaid Source (Whole RAG)

```mermaid
flowchart LR
   subgraph Ingestion
      I1[Load raw contracts] --> I2[Parse title/preamble/sections]
      I2 --> I3[Semantic chunking]
      I3 --> I4[Attach metadata]
      I4 --> I5[Embed chunks]
      I5 --> I6[(Chroma Vector DB)]
   end

   subgraph QueryRuntime[Query Runtime / LangGraph]
      Q1[User Query + thread_id] --> Q2[Supervisor / Rewriter]
      Q2 --> Q3[Researcher: hybrid retrieval + optional reranker]
      Q3 --> Q4[Risk Auditor]
      Q4 --> Q5[Legal Guardian]
      Q5 --> Q6{Route decision}
      Q6 -->|pass| Q7[Final Answer + Risk Report]
      Q6 -->|reflect fail| Q8[Correction loop]
      Q8 --> Q3
   end

   I6 --> Q3
   Q7 --> UI[Streamlit: Agents/Graph views]
```

---

## 5) Ingestion Flow and Key Considerations

### Ingestion Steps

1. Read each raw `.txt` contract.
2. Extract:
   - document title
   - preamble (intro before numbered sections)
   - numbered sections
3. Build semantic chunks while preserving legal context in chunk text:
   - `Title: ...`
   - `Section: ...`
   - `Content: ...`
4. Attach metadata:
   - `source`, `file_name`, `document_title`, `section_number`, `section_header`, `chunk_index`
5. Embed chunks and persist into Chroma.

### Key Things Considered

- **Structure preservation** improves legal retrieval quality.
- **Section-level metadata** supports filtering and explainability.
- **Config-driven behavior** controls model, chunking, paths, and vector space from YAML.
- **Optional reset** allows full reindex when embedding model or vector settings change.

### Ingestion Flow Chart

![Mermaid flowchart illustrating the ingestion flow](docs/images/ingestion_flow_chart.png)                                       

### Mermaid Source (Ingestion)

```mermaid
flowchart TD
   A[Raw .txt files] --> B[Read contract text]
   B --> C[Extract title + preamble + numbered sections]
   C --> D[Build section-aware chunks]
   D --> E[Construct chunk text: Title / Section / Content]
   E --> F[Attach metadata]
   F --> G[Generate embeddings]
   G --> H[(Chroma collection)]

   F -. metadata fields .-> M[source, file_name, document_title, section_number, section_header, chunk_index]
   H -. vector metric .-> V[hnsw:space from config]
```

---

## 6) Streamlit App Capabilities

The Streamlit app (`streamlit_app.py`) provides module-wise testing:

### Ingestion Panel

- Reads config and displays ingestion settings.
- Parses selected contract and shows sections.
- Previews semantic chunks and metadata.
- Runs full ingestion into Chroma.

### Agents Panel

- Step-by-step execution:
  1. Supervisor/Rewriter
  2. Researcher
   3. Risk Auditor
   4. Legal Guardian
- One-click full pipeline execution.
- Supports runtime reflection toggle and max reflection count.
- Shows full state snapshot, retrieved chunks, confidence, warnings, guardian scores, and risk report.

### Graph Panel

- Runs full LangGraph flow with `thread_id`.
- Uses checkpointed memory for multi-turn behavior.
- Includes conditional routing through Legal Guardian and optional reflection loops.
- Shows final state, retrieved chunks, guardian decisions, and risk report.

### Evaluation Panel

- Placeholder for future evaluator integration.

---

## 7) Configuration Summary

Primary config file: `src/resources/config-local.yaml`

Sections:

- `llm`: provider, base URL, model, generation params, optional headers.
- `embeddings`: provider, base URL, embedding model, optional headers.
- `ingestion`: raw path, persist path, collection name, vector space, chunk params, reset flag.
- `guardian`: mode (`off | evaluate_only | reflect`), thresholds, reflection limits.
- `retrieval`: hybrid retrieval weights, top-k, reranker toggle/model controls.

---

## 8) Suggested Run Sequence

1. Validate config and API key.
2. Run ingestion from Streamlit Ingestion panel.
3. Verify retrieved chunks in Agents panel.
4. Run Graph panel for end-to-end query testing.

---

## 9) Future Improvements

- Evaluation metrics dashboard in Streamlit.
- Per-node latency and token telemetry.
- Automatic retrieval/guardian threshold tuning from eval datasets.
