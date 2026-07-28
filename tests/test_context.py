"""
Unit tests for core/rag/context.py

Covers build_context: formats retrieved citations into numbered LLM
prompt blocks.

Moved here from tests/test_pipeline.py when _build_context was extracted
out of pipeline.py into its own module (core/rag/context.py) so it could
be shared with the LangGraph generate() node without reaching into
pipeline.py's private functions.

No OpenAI calls, no ChromaDB, no network — all pure Python.
"""

from __future__ import annotations

from core.rag.context import build_context


class TestBuildContext:
    def test_empty_citations_returns_empty_string(self):
        assert build_context([]) == ""

    def test_single_citation_format(self, single_citation):
        result = build_context(single_citation)
        # Must contain the numbered block marker
        assert result.startswith("[1]")
        # Must embed doc_id and page
        assert "doc_abc123" in result
        assert "p.3" in result
        # Must include the text itself
        assert "Hand hygiene" in result

    def test_multiple_citations_numbered_sequentially(self, multi_citations):
        result = build_context(multi_citations)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_blocks_separated_by_double_newline(self, multi_citations):
        result = build_context(multi_citations)
        # Each block separated by "\n\n"
        blocks = result.split("\n\n")
        assert len(blocks) == 3

    def test_each_citation_contains_its_text(self, multi_citations):
        result = build_context(multi_citations)
        assert "Wash hands with soap" in result
        assert "Alcohol-based hand rubs" in result
        assert "Use PPE" in result

    def test_cross_doc_citations_both_appear(self, multi_citations):
        result = build_context(multi_citations)
        assert "doc_abc123" in result
        assert "doc_xyz999" in result

    def test_missing_meta_fields_do_not_raise(self):
        """build_context must not crash when doc_id or page is absent."""
        citations = [
            {
                "text": "Some guideline text.",
                "meta": {},  # no doc_id, no page
                "distance": 0.15,
            }
        ]
        result = build_context(citations)
        assert "[1]" in result
        assert "Some guideline text." in result

    def test_output_is_string(self, multi_citations):
        result = build_context(multi_citations)
        assert isinstance(result, str)
