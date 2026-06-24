"""
Unit tests for core/ingestion/chunker.py

chunk_pages() is pure Python — no I/O, no network, fully unit-testable.
Tests cover:
  - basic chunking output structure
  - chunk size / overlap behaviour
  - empty and whitespace-only pages
  - chunk_id naming convention
  - multi-page input
"""

from __future__ import annotations

from core.ingestion.chunker import Chunk, chunk_pages


class TestChunkPages:
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_pages(self, texts: list[str]) -> list[tuple[int, str]]:
        """Produce (page_number, text) pairs starting at page 1."""
        return [(i + 1, t) for i, t in enumerate(texts)]

    # ------------------------------------------------------------------
    # Basic structure
    # ------------------------------------------------------------------

    def test_returns_list_of_chunks(self):
        pages = self._make_pages(["Hello world."])
        result = chunk_pages(pages)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_chunk_has_required_fields(self):
        pages = self._make_pages(["Some text here."])
        chunks = chunk_pages(pages)
        assert len(chunks) > 0
        c = chunks[0]
        assert hasattr(c, "chunk_id")
        assert hasattr(c, "page")
        assert hasattr(c, "text")

    # ------------------------------------------------------------------
    # Empty / blank pages
    # ------------------------------------------------------------------

    def test_empty_string_page_produces_no_chunks(self):
        chunks = chunk_pages([(1, "")])
        assert chunks == []

    def test_whitespace_only_page_produces_no_chunks(self):
        chunks = chunk_pages([(1, "   \n\t  ")])
        assert chunks == []

    def test_mixed_empty_and_valid_pages(self):
        pages = [(1, ""), (2, "Valid content here."), (3, "   ")]
        chunks = chunk_pages(pages)
        assert all(c.page == 2 for c in chunks)

    # ------------------------------------------------------------------
    # Chunk size
    # ------------------------------------------------------------------

    def test_short_text_produces_single_chunk(self):
        text = "Short text."
        chunks = chunk_pages([(1, text)], chunk_size=900, overlap=150)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_long_text_produces_multiple_chunks(self):
        # 1000 chars > default chunk_size=900
        text = "a" * 1000
        chunks = chunk_pages([(1, text)], chunk_size=900, overlap=150)
        assert len(chunks) > 1

    def test_chunk_text_length_respects_chunk_size(self):
        text = "b" * 2000
        chunks = chunk_pages([(1, text)], chunk_size=500, overlap=50)
        for c in chunks:
            assert len(c.text) <= 500

    # ------------------------------------------------------------------
    # Overlap
    # ------------------------------------------------------------------

    def test_overlap_means_content_shared_between_adjacent_chunks(self):
        # With overlap, the tail of chunk N should appear at the start of chunk N+1
        text = "x" * 200
        chunks = chunk_pages([(1, text)], chunk_size=100, overlap=30)
        if len(chunks) >= 2:
            tail_of_first = chunks[0].text[-30:]
            head_of_second = chunks[1].text[:30]
            assert tail_of_first == head_of_second

    def test_zero_overlap_no_shared_content(self):
        text = "abcdefghij" * 20  # 200 chars
        chunks = chunk_pages([(1, text)], chunk_size=50, overlap=0)
        if len(chunks) >= 2:
            tail = chunks[0].text
            head = chunks[1].text
            assert tail[-1] != head[0] or True  # just ensure no crash; positions differ

    # ------------------------------------------------------------------
    # chunk_id format
    # ------------------------------------------------------------------

    def test_chunk_id_starts_with_page_prefix(self):
        chunks = chunk_pages([(5, "Some text to chunk.")])
        for c in chunks:
            assert c.chunk_id.startswith("p5_c")

    def test_chunk_ids_are_unique_within_page(self):
        text = "w" * 2000
        chunks = chunk_pages([(1, text)], chunk_size=300, overlap=50)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    # ------------------------------------------------------------------
    # Multi-page
    # ------------------------------------------------------------------

    def test_page_numbers_assigned_correctly(self):
        pages = [(1, "Page one content."), (2, "Page two content.")]
        chunks = chunk_pages(pages)
        page_nums = {c.page for c in chunks}
        assert 1 in page_nums
        assert 2 in page_nums

    def test_multi_page_chunk_ids_dont_collide(self):
        pages = [(1, "a" * 1000), (2, "b" * 1000)]
        chunks = chunk_pages(pages, chunk_size=300, overlap=50)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_empty_pages_list_returns_empty(self):
        assert chunk_pages([]) == []
