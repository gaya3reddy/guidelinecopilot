"""
Unit tests for core/rag/pipeline.py

Covers the remaining pure helper function:
  - _merge_and_topk  : merges per-doc result lists and returns top-k by distance

build_context's tests moved to tests/test_context.py when that function
was extracted out of this module into core/rag/context.py.

No OpenAI calls, no ChromaDB, no network — all pure Python.
"""

from __future__ import annotations

from core.rag.pipeline import _merge_and_topk


# ---------------------------------------------------------------------------
# _merge_and_topk
# ---------------------------------------------------------------------------


class TestMergeAndTopk:
    def _make_result(self, doc_id: str, distance: float) -> dict:
        return {
            "text": f"Text from {doc_id}",
            "meta": {"doc_id": doc_id, "page": 1, "chunk_id": "p1_c0"},
            "distance": distance,
        }

    def test_single_list_passthrough(self):
        results = [[self._make_result("doc_a", 0.1), self._make_result("doc_a", 0.3)]]
        merged = _merge_and_topk(results, top_k=2)
        assert len(merged) == 2

    def test_top_k_limits_output(self):
        results = [
            [self._make_result("doc_a", 0.1), self._make_result("doc_a", 0.5)],
            [self._make_result("doc_b", 0.2), self._make_result("doc_b", 0.6)],
        ]
        merged = _merge_and_topk(results, top_k=2)
        assert len(merged) == 2

    def test_sorted_by_distance_ascending(self):
        results = [
            [self._make_result("doc_a", 0.9), self._make_result("doc_a", 0.1)],
            [self._make_result("doc_b", 0.5), self._make_result("doc_b", 0.3)],
        ]
        merged = _merge_and_topk(results, top_k=4)
        distances = [r["distance"] for r in merged]
        assert distances == sorted(distances)

    def test_best_result_is_first(self):
        results = [
            [self._make_result("doc_a", 0.8)],
            [self._make_result("doc_b", 0.05)],
        ]
        merged = _merge_and_topk(results, top_k=2)
        assert merged[0]["distance"] == 0.05

    def test_empty_sublists_handled(self):
        results = [[], [self._make_result("doc_a", 0.2)], []]
        merged = _merge_and_topk(results, top_k=5)
        assert len(merged) == 1

    def test_empty_input_returns_empty(self):
        merged = _merge_and_topk([], top_k=5)
        assert merged == []

    def test_top_k_larger_than_total_returns_all(self):
        results = [
            [self._make_result("doc_a", 0.1)],
            [self._make_result("doc_b", 0.2)],
        ]
        merged = _merge_and_topk(results, top_k=100)
        assert len(merged) == 2
