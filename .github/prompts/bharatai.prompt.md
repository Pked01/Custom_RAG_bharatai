---
name: bharatai
description: Describe when to use this prompt
---
Project Name: Custom_RAG_bharatai
Goal: Build a multi-agent, local-first RAG system to analyze legal contracts (NDAs, VSAs, SLAs, DPAs) with high precision, multi-turn conversation support, and risk identification.

Core Architecture Requirements (Non-Negotiable):

Orchestration: Use LangGraph to build a state-machine based workflow. State must be explicitly managed via a TypedDict to track message history, retrieved context, identified risks, and the current document in focus.

Session Persistence: Implement Checkpointers (e.g., MemorySaver or SqliteSaver) to handle multi-turn conversations and distinct user threads.

Vector Storage: Use ChromaDB for local vector persistence. Implement a script that chunks documents based on structural markers (Sections/Articles) and stores them with metadata (document name, section ID).

Agent Strategy: Separate responsibilities into "meaningful" agents:

Supervisor (The Router): Analyzes user intent and routes to specialists.

Researcher (The Extractor): Performs RAG to find specific clauses. MUST provide exact citations and page/section references.

Risk Auditor (The Critic): Analyzes the Researcher's output to identify legal liabilities, missing protections, or conflicts across agreements.

Proposed Folder Structure:

Plaintext
Custom_RAG_bharatai/
├── data/
│   ├── raw/                 # Source .txt/.pdf files
│   └── chroma_db/           # Persistent Vector DB storage
├── src/
│   ├── agents/              # Logic for Supervisor, Researcher, Auditor
│   ├── graph/               # LangGraph state and workflow definitions
│   ├── ingestion/           # Chunking and embedding logic
│   └── utils/               # Config (YAML/ENV) and Logging
├── main.py                  # CLI-based interactive entry point
├── requirements.txt
└── README.md
Legal Reasoning Context:

The system must prioritize Grounding. If a clause isn't found, the agent should say "Not found" rather than hallucinate.

Support "Multi-Hop" queries where a clause in one document (e.g., DPA) refers to a limit in another (e.g., VSA).

Every answer must include a Risk Flag (Low, Medium, High) if the query pertains to liability or financial exposure.

Next Step: Help me initialize the src/graph/state.py file with a robust AgentState definition and the main.py entry point that supports a multi-turn CLI loop using LangGraph threads.