"""
retrieve node — wraps the existing HybridRetriever (BM25 + vector + RRF +
reranker) so it can plug into the LangGraph state machine.

Deliberately mirrors the retriever setup in core/rag/pipeline.py's
answer_question() rather than introducing a new construction pattern.
No retrieval logic is duplicated or reimplemented here — this is purely
an adapter between GraphState and the retriever that's already tested
and RAGAS-validated.
"""

from __future__ import annotations

from apps.api.config import settings
from core.graph.graph_state import GraphState
from core.retrieval.bm25_store import BM25Store
from core.retrieval.embedder import OpenAIEmbedder
from core.retrieval.hybrid import HybridRetriever
from core.retrieval.vectorstore import ChromaVectorStore


def retrieve(state: GraphState) -> dict:
    """Run hybrid search + rerank for the current query, write results to state.

    Reads: state["question"] (the current, possibly-rewritten query)
    Writes: state["documents"]

    Note: builds the retriever fresh on every call, same as
    answer_question() does today. Left as-is intentionally — this node
    is an adapter, not a place to change retriever lifecycle.
    """
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY missing. Set it in .env.")

    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key, model=settings.openai_embed_model
    )
    store = ChromaVectorStore(
        persist_dir=str(settings.processed_dir / "chroma"), embedder=embedder
    )
    bm25 = BM25Store(store.get_all_chunks())
    retriever = HybridRetriever(vector_store=store, bm25_store=bm25)

    documents = retriever.search(query=state["question"], top_k=5, doc_id=None)

    return {"documents": documents}
