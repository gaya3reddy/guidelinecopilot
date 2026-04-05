from __future__ import annotations

import time
import uuid
from fastapi import APIRouter, HTTPException

from apps.api.config import settings
from core.schemas.models import AskRequest, AskResponse, Meta, Citation
from core.rag.pipeline import answer_question
from fastapi.responses import StreamingResponse
from core.rag.pipeline import stream_answer
from core.schemas.utils import distance_to_score

router = APIRouter(tags=["rag"])


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

    citations: list[Citation] = []
    for c in out.get("citations", []):
        meta = c["meta"]
        text = c["text"]
        dist = float(c["distance"])

        citations.append(
            Citation(
                doc_id=str(meta.get("doc_id", "")),
                page=int(meta.get("page") or 0),
                chunk_id=str(meta.get("chunk_id", "")),
                snippet=text[:350],  # keep UI readable
                score=distance_to_score(dist),
            )
        )

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
