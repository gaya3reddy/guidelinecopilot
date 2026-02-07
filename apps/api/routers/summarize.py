from __future__ import annotations

import time
import uuid
from fastapi import APIRouter, HTTPException

from apps.api.config import settings
from core.schemas.models import SummarizeRequest, SummarizeResponse, Meta, Citation
from core.rag.pipeline import summarize_guideline

router = APIRouter(tags=["summarize"])


def _distance_to_score(distance: float) -> float:
    s = 1.0 / (1.0 + max(0.0, float(distance)))
    return max(0.0, min(1.0, s))


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest) -> SummarizeResponse:
    start = time.perf_counter()
    request_id = f"req_{uuid.uuid4().hex[:10]}"

    try:
        out = summarize_guideline(
            style=req.style,
            doc_ids=req.doc_ids,
            top_k=6,     # keep stable for now-> reduced to8 to 6 to reduce latency and cost, since we do an extra round of re-ranking in the prompt
            mode="rag",  # stable Day-4 scope
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
                snippet=text[:350],
                score=_distance_to_score(dist),
            )
        )

    latency_ms = int((time.perf_counter() - start) * 1000)
    meta = Meta(
        request_id=request_id,
        latency_ms=latency_ms,
        model=settings.openai_chat_model if settings.model_provider == "openai" else settings.model_provider,
        prompt_version="summarize_v1",
    )
    return SummarizeResponse(summary=out["summary"], citations=citations, meta=meta)
