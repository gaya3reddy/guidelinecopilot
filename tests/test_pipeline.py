"""
Unit tests for core/rag/pipeline.py

Covers the two pure helper functions:
  - _build_context   : formats retrieved citations into numbered LLM prompt blocks
  - _merge_and_topk  : merges per-doc result lists and returns top-k by distance

No OpenAI calls, no ChromaDB, no network — all pure Python.
"""

from __future__ import annotations

from core.rag.pipeline import _build_context, _merge_and_topk


# ---------------------------------------------------------------------------
# _build_context
# ---------------------------------------------------------------------------


class TestBuildContext:
    def test_empty_citations_returns_empty_string(self):
        assert _build_context([]) == ""

    def test_single_citation_format(self, single_citation):
        result = _build_context(single_citation)
        # Must contain the numbered block marker
        assert result.startswith("[1]")
        # Must embed doc_id and page
        assert "doc_abc123" in result
        assert "p.3" in result
        # Must include the text itself
        assert "Hand hygiene" in result

    def test_multiple_citations_numbered_sequentially(self, multi_citations):
        result = _build_context(multi_citations)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_blocks_separated_by_double_newline(self, multi_citations):
        result = _build_context(multi_citations)
        # Each block separated by "\n\n"
        blocks = result.split("\n\n")
        assert len(blocks) == 3

    def test_each_citation_contains_its_text(self, multi_citations):
        result = _build_context(multi_citations)
        assert "Wash hands with soap" in result
        assert "Alcohol-based hand rubs" in result
        assert "Use PPE" in result

    def test_cross_doc_citations_both_appear(self, multi_citations):
        result = _build_context(multi_citations)
        assert "doc_abc123" in result
        assert "doc_xyz999" in result

    def test_missing_meta_fields_do_not_raise(self):
        """_build_context must not crash when doc_id or page is absent."""
        citations = [
            {
                "text": "Some guideline text.",
                "meta": {},  # no doc_id, no page
                "distance": 0.15,
            }
        ]
        result = _build_context(citations)
        assert "[1]" in result
        assert "Some guideline text." in result

    def test_output_is_string(self, multi_citations):
        result = _build_context(multi_citations)
        assert isinstance(result, str)


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
