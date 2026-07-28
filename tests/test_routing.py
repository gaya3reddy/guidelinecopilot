"""
Unit tests for core/graph/routing.py

Covers decide_to_generate: the conditional edge function that routes
between "generate" and "rewrite_query" after grade_documents runs.

Pure logic, no LLM calls, no mocking needed — this function only reads
state["grade"] and state["retry_count"].
"""

from __future__ import annotations

from core.graph.graph_state import MAX_RETRIES, GraphState
from core.graph.routing import decide_to_generate


def _make_state(grade: str, retry_count: int) -> GraphState:
    """Minimal state — decide_to_generate only reads grade and retry_count,
    so the other fields are irrelevant filler."""
    return GraphState(
        question="q",
        original_question="q",
        documents=[],
        generation="",
        retry_count=retry_count,
        grade=grade,
    )


class TestDecideToGenerate:
    def test_relevant_grade_routes_to_generate(self):
        state = _make_state(grade="yes", retry_count=0)
        assert decide_to_generate(state) == "generate"

    def test_irrelevant_grade_with_retries_left_routes_to_rewrite(self):
        state = _make_state(grade="no", retry_count=0)
        assert decide_to_generate(state) == "rewrite_query"

    def test_irrelevant_grade_below_max_retries_still_rewrites(self):
        state = _make_state(grade="no", retry_count=MAX_RETRIES - 1)
        assert decide_to_generate(state) == "rewrite_query"

    def test_irrelevant_grade_at_max_retries_falls_through_to_generate(self):
        """Retry cap takes priority over the grade — this is the graceful
        degradation path that prevents infinite loops."""
        state = _make_state(grade="no", retry_count=MAX_RETRIES)
        assert decide_to_generate(state) == "generate"

    def test_irrelevant_grade_beyond_max_retries_still_generates(self):
        """Defensive: even if retry_count somehow exceeds the cap, we
        should never route to rewrite_query again."""
        state = _make_state(grade="no", retry_count=MAX_RETRIES + 5)
        assert decide_to_generate(state) == "generate"

    def test_relevant_grade_at_max_retries_still_routes_to_generate(self):
        """A "yes" grade always wins regardless of retry_count — the cap
        only matters on the "no" path."""
        state = _make_state(grade="yes", retry_count=MAX_RETRIES)
        assert decide_to_generate(state) == "generate"
