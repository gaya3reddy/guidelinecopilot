"""
Unit tests for core/graph/nodes/grade_documents.py

Mocks ChatOpenAI entirely — no OpenAI calls, no network. Follows the same
monkeypatch.setattr pattern used in tests/test_reranker.py: patch the name
as it's bound in the target module's namespace (grade_documents.py does
`from langchain_openai import ChatOpenAI` and `from apps.api.config import
settings` at module level, so both names live in
core.graph.nodes.grade_documents's namespace, not their original modules).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from core.graph.graph_state import GraphState
from core.graph.nodes.grade_documents import GradeDocuments, grade_documents


def _make_state(documents: list[dict], question: str = "What is X?") -> GraphState:
    return GraphState(
        question=question,
        original_question=question,
        documents=documents,
        generation="",
        retry_count=0,
        grade="",
    )


def _mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.openai_chat_model = "gpt-4o-mini"
    return settings


def _mock_chat_openai(binary_score: str) -> MagicMock:
    """Build a mock ChatOpenAI class whose .with_structured_output(...).invoke(...)
    returns a GradeDocuments with the given score."""
    mock_grader = MagicMock()
    mock_grader.invoke.return_value = GradeDocuments(binary_score=binary_score)

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_grader

    mock_chat_cls = MagicMock(return_value=mock_llm)
    return mock_chat_cls, mock_llm, mock_grader


class TestGradeDocuments:
    def test_empty_documents_short_circuits_to_no(self, monkeypatch):
        """No documents means no LLM call at all — cheap guard, not a
        classification decision."""
        mock_chat_cls = MagicMock()
        monkeypatch.setattr(
            "core.graph.nodes.grade_documents.ChatOpenAI", mock_chat_cls
        )
        monkeypatch.setattr(
            "core.graph.nodes.grade_documents.settings", _mock_settings()
        )

        result = grade_documents(_make_state(documents=[]))

        assert result == {"grade": "no"}
        mock_chat_cls.assert_not_called()

    def test_relevant_documents_score_yes(self, monkeypatch, single_citation):
        mock_chat_cls, _, _ = _mock_chat_openai(binary_score="yes")
        monkeypatch.setattr(
            "core.graph.nodes.grade_documents.ChatOpenAI", mock_chat_cls
        )
        monkeypatch.setattr(
            "core.graph.nodes.grade_documents.settings", _mock_settings()
        )

        result = grade_documents(_make_state(documents=single_citation))

        assert result == {"grade": "yes"}

    def test_irrelevant_documents_score_no(self, monkeypatch, single_citation):
        mock_chat_cls, _, _ = _mock_chat_openai(binary_score="no")
        monkeypatch.setattr(
            "core.graph.nodes.grade_documents.ChatOpenAI", mock_chat_cls
        )
        monkeypatch.setattr(
            "core.graph.nodes.grade_documents.settings", _mock_settings()
        )

        result = grade_documents(_make_state(documents=single_citation))

        assert result == {"grade": "no"}

    def test_llm_constructed_with_configured_chat_model(
        self, monkeypatch, single_citation
    ):
        mock_chat_cls, _, _ = _mock_chat_openai(binary_score="yes")
        monkeypatch.setattr(
            "core.graph.nodes.grade_documents.ChatOpenAI", mock_chat_cls
        )
        monkeypatch.setattr(
            "core.graph.nodes.grade_documents.settings", _mock_settings()
        )

        grade_documents(_make_state(documents=single_citation))

        mock_chat_cls.assert_called_once_with(model="gpt-4o-mini", temperature=0)

    def test_grader_prompt_includes_question_and_excerpt_text(
        self, monkeypatch, single_citation
    ):
        """The grader has to actually see the question and the retrieved
        text to judge relevance — a regression here (e.g. only passing
        the question) would silently make grading meaningless."""
        mock_chat_cls, _, mock_grader = _mock_chat_openai(binary_score="yes")
        monkeypatch.setattr(
            "core.graph.nodes.grade_documents.ChatOpenAI", mock_chat_cls
        )
        monkeypatch.setattr(
            "core.graph.nodes.grade_documents.settings", _mock_settings()
        )

        state = _make_state(documents=single_citation, question="What is hand hygiene?")
        grade_documents(state)

        sent_messages = mock_grader.invoke.call_args[0][0]
        user_content = next(m["content"] for m in sent_messages if m["role"] == "user")
        assert "What is hand hygiene?" in user_content
        assert single_citation[0]["text"] in user_content
