from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI

from apps.api.config import settings
from core.rag.prompts import ASK_SYSTEM
from core.retrieval.embedder import OpenAIEmbedder
from core.retrieval.vectorstore import ChromaVectorStore


def _build_context(citations: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, c in enumerate(citations, start=1):
        meta = c["meta"]
        doc_id = meta.get("doc_id")
        page = meta.get("page")
        text = c["text"]
        blocks.append(f"[{i}] ({doc_id} p.{page})\n{text}")
    return "\n\n".join(blocks)


def _merge_and_topk(results: list[list[dict]], top_k: int) -> list[dict]:
    merged = []
    for r in results:
        merged.extend(r)
    merged.sort(key=lambda x: x["distance"])  # smaller distance = better
    return merged[:top_k]


def answer_question(
    question: str,
    top_k: int = 5,
    doc_ids: list[str] | None = None,
    mode: str = "rag",
) -> Dict[str, Any]:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY missing. Set it in .env.")

    embedder = OpenAIEmbedder(api_key=settings.openai_api_key, model=settings.openai_embed_model)
    store = ChromaVectorStore(persist_dir=str(settings.processed_dir / "chroma"), embedder=embedder)

    # --- NEW: retrieval logic ---
    retrieved: list[dict] = []
    if mode == "no_rag":
        retrieved = []
    else:
        doc_ids = doc_ids or []
        if len(doc_ids) == 0:
            retrieved = store.query(question=question, top_k=top_k, doc_id=None)
        elif len(doc_ids) == 1:
            retrieved = store.query(question=question, top_k=top_k, doc_id=doc_ids[0])
        else:
            per_doc = [store.query(question=question, top_k=top_k, doc_id=did) for did in doc_ids]
            retrieved = _merge_and_topk(per_doc, top_k=top_k)

    context = _build_context(retrieved)
    user_prompt = f"""Question: {question}

Guideline excerpts:
{context}
"""

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
    return {"answer": answer, "citations": retrieved}
