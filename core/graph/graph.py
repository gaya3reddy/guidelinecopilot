"""
Builds the agentic retrieval graph: hybrid search -> grade -> either
generate or rewrite-and-retry.

    START -> retrieve -> grade_documents --(yes)--> generate -> END
                 ^                |
                 |            (no, retries left)
                 +---- rewrite_query

decide_to_generate (core/graph/routing.py) is the conditional edge that
picks between "generate" and "rewrite_query" after grade_documents runs.
It also caps retries via MAX_RETRIES, so this loop always terminates.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from core.graph.graph_state import GraphState
from core.graph.nodes.generate import generate
from core.graph.nodes.grade_documents import grade_documents
from core.graph.nodes.retrieve import retrieve
from core.graph.nodes.rewrite_query import rewrite_query
from core.graph.routing import decide_to_generate


def build_graph():
    """Compile the agentic retrieval graph. Call once; reuse the compiled app."""
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()


def build_initial_state(question: str) -> GraphState:
    """Construct the starting GraphState for a fresh question."""
    return GraphState(
        question=question,
        original_question=question,
        documents=[],
        generation="",
        retry_count=0,
        grade="",
    )
