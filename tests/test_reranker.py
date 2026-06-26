"""
Unit tests for core/retrieval/reranker.py and the _maybe_rerank helper in
core/retrieval/hybrid.py.

No model download, no ChromaDB, no OpenAI — fully mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunks(*texts: str) -> list[dict]:
    """Build minimal chunk dicts matching the pipeline's standard shape."""
    return [
        {
            "text": t,
            "meta": {"doc_id": "doc_test", "chunk_id": f"c{i}", "page": i + 1},
            "distance": 0.1 * i,
            "bm25_score": 1.0,
            "rrf_score": 1.0 / (60 + i + 1),
        }
        for i, t in enumerate(texts)
    ]


def _mock_predict(pairs, **_kwargs):
    """
    Fake scorer: score = number of query words found in the passage.
    Deterministic, no model required.
    """
    query_words = set(pairs[0][0].lower().split())
    scores = []
    for _, passage in pairs:
        hits = sum(1 for w in query_words if w in passage.lower())
        scores.append(float(hits))
    return np.array(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level cross-encoder singleton between tests."""
    import core.retrieval.reranker as reranker_mod

    reranker_mod._cross_encoder = None
    yield
    reranker_mod._cross_encoder = None


@pytest.fixture()
def mock_model(monkeypatch):
    """Patch _get_model() so no real model is loaded."""
    m = MagicMock()
    m.predict.side_effect = _mock_predict
    monkeypatch.setattr("core.retrieval.reranker._get_model", lambda: m)
    return m


# ---------------------------------------------------------------------------
# rerank() tests
# ---------------------------------------------------------------------------


class TestRerank:
    def test_returns_top_k(self, mock_model):
        from core.retrieval.reranker import rerank

        chunks = _make_chunks("alpha beta", "gamma", "alpha", "beta gamma delta")
        result = rerank("alpha beta", chunks, top_k=2)

        assert len(result) == 2

    def test_orders_by_relevance(self, mock_model):
        """Most-relevant chunk should come first."""
        from core.retrieval.reranker import rerank

        chunks = _make_chunks(
            "irrelevant text about nothing",  # 0 query word hits
            "ethanol guideline",  # 1 hit
            "80% v/v ethanol concentration",  # 2 hits — best
        )
        result = rerank("ethanol concentration", chunks, top_k=3)

        assert result[0]["text"] == "80% v/v ethanol concentration"

    def test_empty_input_returns_empty(self, mock_model):
        from core.retrieval.reranker import rerank

        assert rerank("anything", [], top_k=5) == []

    def test_top_k_larger_than_chunks(self, mock_model):
        """top_k > len(chunks) should return all chunks, not error."""
        from core.retrieval.reranker import rerank

        chunks = _make_chunks("a", "b")
        result = rerank("a b", chunks, top_k=10)

        assert len(result) == 2

    def test_chunk_fields_preserved(self, mock_model):
        """rerank() must not strip meta, distance, rrf_score etc."""
        from core.retrieval.reranker import rerank

        chunks = _make_chunks("guideline text")
        result = rerank("guideline", chunks, top_k=1)

        assert result[0]["meta"]["doc_id"] == "doc_test"
        assert "distance" in result[0]
        assert "rrf_score" in result[0]

    def test_missing_text_key_does_not_crash(self, mock_model):
        """Chunks without 'text' fall back to empty string, not KeyError."""
        from core.retrieval.reranker import rerank

        chunks = [{"meta": {"doc_id": "x", "chunk_id": "c0"}}]
        result = rerank("anything", chunks, top_k=1)

        assert len(result) == 1


# ---------------------------------------------------------------------------
# _maybe_rerank() tests
# ---------------------------------------------------------------------------


class TestMaybeRerank:
    def test_disabled_returns_rrf_order(self, monkeypatch):
        """When reranker_enabled=False, RRF slice is returned unchanged."""
        mock_settings = MagicMock()
        mock_settings.reranker_enabled = False
        # _maybe_rerank does `from apps.api.config import settings` at call time,
        # so patch the source module, not hybrid.
        monkeypatch.setattr("apps.api.config.settings", mock_settings)

        from core.retrieval.hybrid import _maybe_rerank

        chunks = _make_chunks("first", "second", "third")
        result = _maybe_rerank("query", chunks, top_k=2)

        assert result == chunks[:2]

    def test_enabled_calls_reranker(self, monkeypatch, mock_model):
        """When reranker_enabled=True, rerank() is called."""
        mock_settings = MagicMock()
        mock_settings.reranker_enabled = True
        mock_settings.reranker_top_k = 2
        monkeypatch.setattr("apps.api.config.settings", mock_settings)

        def _fake_rerank(query, chunks, top_k):
            return chunks[:top_k]

        monkeypatch.setattr("core.retrieval.reranker.rerank", _fake_rerank)

        from core.retrieval.hybrid import _maybe_rerank

        chunks = _make_chunks("a", "b", "c")
        result = _maybe_rerank("query", chunks, top_k=2)

        assert len(result) == 2

    def test_fallback_on_reranker_error(self, monkeypatch):
        """If reranker raises, fall back to RRF slice without crashing."""
        mock_settings = MagicMock()
        mock_settings.reranker_enabled = True
        mock_settings.reranker_top_k = 2
        monkeypatch.setattr("apps.api.config.settings", mock_settings)

        def _bad_rerank(*args, **kwargs):
            raise RuntimeError("model load failed")

        monkeypatch.setattr("core.retrieval.reranker.rerank", _bad_rerank)

        from core.retrieval.hybrid import _maybe_rerank

        chunks = _make_chunks("a", "b", "c")
        result = _maybe_rerank("query", chunks, top_k=2)

        assert len(result) == 2
        assert result[0]["text"] == "a"
