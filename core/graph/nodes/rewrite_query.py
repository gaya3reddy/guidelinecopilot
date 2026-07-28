"""
rewrite_query node — reformulates the search query after grade_documents
flags the current retrieval as insufficient.

Reads original_question (not question) as the source to rewrite from,
so repeated rewrites don't compound drift from an earlier, already-narrowed
attempt. Increments retry_count so the routing function can cap the loop.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from apps.api.config import settings
from core.graph.graph_state import GraphState

REWRITE_SYSTEM = """You rewrite search queries for a clinical guideline \
retrieval system. The previous query did not retrieve relevant excerpts.

Produce a single reformulated query that is more likely to match relevant \
text in the guideline. You may:
- expand abbreviations or add clinical synonyms
- rephrase from a question into a keyword/topic style query
- broaden an overly narrow query, or narrow an overly broad one

Return ONLY the rewritten query text — no explanation, no quotes."""


def rewrite_query(state: GraphState) -> dict:
    """Reformulate the query for another retrieval attempt.

    Reads: state["original_question"]
    Writes: state["question"] (overwritten with the new query),
            state["retry_count"] (incremented)
    """
    llm = ChatOpenAI(model=settings.openai_chat_model, temperature=0.3)

    result = llm.invoke(
        [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": state["original_question"]},
        ]
    )

    return {
        "question": result.content.strip(),
        "retry_count": state["retry_count"] + 1,
    }
