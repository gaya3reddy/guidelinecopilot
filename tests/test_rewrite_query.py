"""
Unit tests for core/graph/nodes/rewrite_query.py

Mocks ChatOpenAI entirely — no OpenAI calls, no network.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from core.graph.graph_state import GraphState
from core.graph.nodes.rewrite_query import rewrite_query


def _make_state(
    question: str, original_question: str, retry_count: int = 0
) -> GraphState:
    return GraphState(
        question=question,
        original_question=original_question,
        documents=[],
        generation="",
        retry_count=retry_count,
        grade="no",
    )


def _mock_chat_openai(response_content: str):
    mock_response = MagicMock()
    mock_response.content = response_content

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    mock_chat_cls = MagicMock(return_value=mock_llm)
    return mock_chat_cls, mock_llm


def _mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.openai_chat_model = "gpt-4o-mini"
    return settings


class TestRewriteQuery:
    def test_returns_rewritten_question_stripped(self, monkeypatch):
        mock_chat_cls, _ = _mock_chat_openai("  Antibiotics for pneumonia treatment  ")
        monkeypatch.setattr("core.graph.nodes.rewrite_query.ChatOpenAI", mock_chat_cls)
        monkeypatch.setattr("core.graph.nodes.rewrite_query.settings", _mock_settings())

        state = _make_state(
            question="What antibiotics treat pneumonia?",
            original_question="What antibiotics treat pneumonia?",
        )
        result = rewrite_query(state)

        assert result["question"] == "Antibiotics for pneumonia treatment"

    def test_increments_retry_count(self, monkeypatch):
        mock_chat_cls, _ = _mock_chat_openai("rewritten")
        monkeypatch.setattr("core.graph.nodes.rewrite_query.ChatOpenAI", mock_chat_cls)
        monkeypatch.setattr("core.graph.nodes.rewrite_query.settings", _mock_settings())

        state = _make_state(question="q", original_question="q", retry_count=1)
        result = rewrite_query(state)

        assert result["retry_count"] == 2

    def test_rewrites_from_original_question_not_current_question(self, monkeypatch):
        """On a second rewrite, the source must be original_question, not
        the already-once-rewritten question — otherwise each successive
        rewrite drifts further from what the user actually asked."""
        mock_chat_cls, mock_llm = _mock_chat_openai("second rewrite")
        monkeypatch.setattr("core.graph.nodes.rewrite_query.ChatOpenAI", mock_chat_cls)
        monkeypatch.setattr("core.graph.nodes.rewrite_query.settings", _mock_settings())

        state = _make_state(
            question="already-rewritten intermediate query",
            original_question="What antibiotics treat pneumonia?",
            retry_count=1,
        )
        rewrite_query(state)

        sent_messages = mock_llm.invoke.call_args[0][0]
        user_content = next(m["content"] for m in sent_messages if m["role"] == "user")
        assert user_content == "What antibiotics treat pneumonia?"
        assert user_content != "already-rewritten intermediate query"

    def test_llm_constructed_with_configured_model_and_nonzero_temperature(
        self, monkeypatch
    ):
        """temperature=0.3 (not 0) is deliberate here — unlike the grader,
        rewriting benefits from some variation between attempts rather
        than deterministic output."""
        mock_chat_cls, _ = _mock_chat_openai("rewritten")
        monkeypatch.setattr("core.graph.nodes.rewrite_query.ChatOpenAI", mock_chat_cls)
        monkeypatch.setattr("core.graph.nodes.rewrite_query.settings", _mock_settings())

        state = _make_state(question="q", original_question="q")
        rewrite_query(state)

        mock_chat_cls.assert_called_once_with(model="gpt-4o-mini", temperature=0.3)
