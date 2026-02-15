from __future__ import annotations

# What this file does:
# - Builds a complete multi-turn graph: query_rewriter -> researcher -> risk_auditor.
# - Uses MemorySaver checkpointing for session continuity with thread_id.
#
# Why this approach:
# - Keeps transitions explicit and easy to test.
# - Avoids open-ended control flow by routing to END when clarification is needed.

from functools import lru_cache

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.legal_guardian import legal_guardian_node, route_after_legal_guardian
from src.agents.researcher import researcher_node
from src.agents.risk_auditor import risk_auditor_node
from src.agents.supervisor import query_rewriter_node, route_after_research
from src.graph.state import AgentState


@lru_cache(maxsize=1)
def build_graph():
	graph = StateGraph(AgentState)

	graph.add_node("query_rewriter", query_rewriter_node)
	graph.add_node("researcher", researcher_node)
	graph.add_node("risk_auditor", risk_auditor_node)
	graph.add_node("legal_guardian", legal_guardian_node)

	graph.add_edge(START, "query_rewriter")
	graph.add_edge("query_rewriter", "researcher")
	graph.add_conditional_edges(
		"researcher",
		route_after_research,
		{
			"risk_auditor": "risk_auditor",
			"END": END,
		},
	)
	graph.add_edge("risk_auditor", "legal_guardian")
	graph.add_conditional_edges(
		"legal_guardian",
		route_after_legal_guardian,
		{
			"researcher": "researcher",
			"END": END,
		},
	)

	return graph.compile(checkpointer=MemorySaver())


def run_graph_once(user_query: str, thread_id: str = "default") -> AgentState:
	app = build_graph()
	initial_state: AgentState = {
		"messages": [HumanMessage(content=user_query)],
		"user_query": user_query,
	}
	return app.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})
