"""
generate node — produces the final answer from the current retrieval.

Deliberately mirrors answer_question()'s generation logic in
core/rag/pipeline.py: same ASK_SYSTEM prompt, same build_context formatting,
same OpenAI call shape. This node does not introduce a different prompt
or a different model-call pattern — the goal is that swapping the old
answer_question() call for graph.invoke() in the API route changes the
orchestration, not the answer quality your RAGAS baseline already validated.
"""

from __future__ import annotations

from openai import OpenAI

from apps.api.config import settings
from core.graph.graph_state import GraphState
from core.rag.context import build_context
from core.rag.prompts import ASK_SYSTEM, ASK_USER_SUFFIX


def generate(state: GraphState) -> dict:
    """Produce the final answer using the current best retrieval.

    Reads: state["documents"], state["original_question"]
    Writes: state["generation"]

    Uses original_question (not question) so the answer addresses what
    the user actually asked, even if retrieval went through one or more
    rewritten queries to get here.
    """
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY missing. Set it in .env.")

    context = build_context(state["documents"])
    user_prompt = (
        f"Question: {state['original_question']}\n\n"
        f"Guideline excerpts:\n{context}{ASK_USER_SUFFIX}"
    )

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": ASK_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer = resp.choices[0].message.content.strip()
    return {"generation": answer}
