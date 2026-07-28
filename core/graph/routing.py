"""
Conditional edge functions for the agentic retrieval graph.

These are NOT nodes — they don't mutate state. LangGraph calls them after
a node runs, and their return value (a string) selects which edge to
follow next. Kept in a separate module from nodes/ since they're a
different kind of thing: pure routing decisions, not state transformations.
"""

from __future__ import annotations

from core.graph.graph_state import MAX_RETRIES, GraphState


def decide_to_generate(state: GraphState) -> str:
    """Route after grade_documents: proceed to generate, or loop to rewrite.

    Retry cap takes priority over the grade — if we've already rewritten
    MAX_RETRIES times, we stop looping regardless of grade and generate
    with whatever we have (generate()'s prompt already handles "not
    covered" via ASK_SYSTEM's existing refusal rule, so this degrades
    gracefully rather than erroring).
    """
    if state["grade"] == "yes":
        return "generate"

    if state["retry_count"] >= MAX_RETRIES:
        return "generate"

    return "rewrite_query"
