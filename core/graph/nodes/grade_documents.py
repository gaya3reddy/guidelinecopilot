"""
grade_documents node — LLM-as-judge that decides whether the current
retrieval is sufficient to answer the question, before we spend a
generation call on it.

This is the node that makes the graph "agentic" rather than a fixed
pipeline: its output feeds a conditional edge that either proceeds to
generate() or loops back to rewrite_query().

Uses langchain-openai's structured output binding (.with_structured_output)
rather than the raw OpenAI client used elsewhere in the pipeline — this is
a deliberate, contained use of a LangChain concept for a task (constrained
yes/no classification) where it earns its keep, without pulling the rest
of the pipeline into LangChain abstractions.
"""

from __future__ import annotations

from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from apps.api.config import settings
from core.graph.graph_state import GraphState

GRADE_SYSTEM = """You are a grader assessing whether retrieved guideline \
excerpts are sufficient to answer a user's question.

Give a binary score:
- "yes" if the excerpts contain information that directly addresses the question.
- "no" if the excerpts are off-topic, too sparse, or do not address the question.

Be strict. Partial or tangential relevance should score "no" — the goal is \
to catch retrieval failures before they reach the answer generation step, \
not to guess at plausibility."""


class GradeDocuments(BaseModel):
    """Structured grading output."""

    binary_score: Literal["yes", "no"] = Field(
        description="Are the retrieved excerpts relevant to the question? 'yes' or 'no'."
    )


def _format_excerpts(documents: list[dict]) -> str:
    """Plain-text rendering of retrieved chunks for the grading prompt.

    Kept local and minimal rather than importing pipeline._build_context —
    this node only needs the raw text, not the citation-formatted version
    generate() will eventually use.
    """
    return "\n\n".join(d["text"] for d in documents)


def grade_documents(state: GraphState) -> dict:
    """Judge retrieval quality; loop-control decision happens downstream.

    Reads: state["documents"], state["original_question"]
    Writes: state["grade"] — "yes" or "no", consumed by the routing
    function (decide_to_generate) to choose the next edge.
    """
    if not state["documents"]:
        return {"grade": "no"}

    llm = ChatOpenAI(model=settings.openai_chat_model, temperature=0)
    grader = llm.with_structured_output(GradeDocuments)

    excerpts = _format_excerpts(state["documents"])
    result: GradeDocuments = grader.invoke(
        [
            {"role": "system", "content": GRADE_SYSTEM},
            {
                "role": "user",
                "content": f"Question: {state['original_question']}\n\n"
                f"Retrieved excerpts:\n{excerpts}",
            },
        ]
    )

    return {"grade": result.binary_score}
