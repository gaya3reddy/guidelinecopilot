from __future__ import annotations

import time
import uuid
from fastapi import APIRouter, HTTPException

from apps.api.config import settings
from core.schemas.models import AskRequest, AskResponse, Meta, Citation
from core.rag.pipeline import answer_question
from fastapi.responses import StreamingResponse
from core.rag.pipeline import stream_answer
from core.schemas.utils import documents_to_citations
from core.graph.graph import build_graph, build_initial_state

router = APIRouter(tags=["rag"])

# Compiled once at import time and reused across requests — building the
# StateGraph is cheap, but there's no reason to redo it per-request.
_agentic_graph = build_graph()


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    start = time.perf_counter()
    request_id = f"req_{uuid.uuid4().hex[:10]}"

    try:
        out = answer_question(
            question=req.question,
            top_k=req.top_k,
            doc_ids=req.doc_ids,  # list[str]
            mode=req.mode,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    citations = [
        Citation(**c) for c in documents_to_citations(out.get("citations", []))
    ]

    latency_ms = int((time.perf_counter() - start) * 1000)
    meta = Meta(
        request_id=request_id,
        latency_ms=latency_ms,
        model=settings.openai_chat_model
        if settings.model_provider == "openai"
        else settings.model_provider,
        prompt_version="ask_v1",
    )
    return AskResponse(answer=out["answer"], citations=citations, meta=meta)


@router.post("/ask/agentic", response_model=AskResponse)
def ask_agentic(req: AskRequest) -> AskResponse:
    """Same contract as /ask, but retrieval runs through the LangGraph
    self-correcting loop (retrieve -> grade -> generate or rewrite-and-retry)
    instead of a single-shot hybrid search.

    doc_ids/mode from AskRequest aren't wired into the graph yet — the
    graph currently always does a corpus-wide search (doc_id=None) and
    mode="rag". Multi-doc filtering can be added to GraphState later if
    needed; not required for the current single-document demo.
    """
    start = time.perf_counter()
    request_id = f"req_{uuid.uuid4().hex[:10]}"

    try:
        initial_state = build_initial_state(req.question)
        final_state = _agentic_graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    citations = [
        Citation(**c) for c in documents_to_citations(final_state.get("documents", []))
    ]

    latency_ms = int((time.perf_counter() - start) * 1000)
    meta = Meta(
        request_id=request_id,
        latency_ms=latency_ms,
        model=settings.openai_chat_model
        if settings.model_provider == "openai"
        else settings.model_provider,
        prompt_version="ask_agentic_v1",
        retries=final_state["retry_count"],
    )
    return AskResponse(answer=final_state["generation"], citations=citations, meta=meta)


@router.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    def generate():
        try:
            yield from stream_answer(
                question=req.question,
                top_k=req.top_k,
                doc_ids=req.doc_ids,
                mode=req.mode,
            )
        except Exception as e:
            yield f"\n\n__ERROR__:{str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")
