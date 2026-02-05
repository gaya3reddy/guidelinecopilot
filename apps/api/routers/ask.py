from __future__ import annotations

import time
import uuid
from fastapi import APIRouter, HTTPException

from apps.api.config import settings
from core.schemas.models import AskRequest, AskResponse, Meta, Citation
from core.rag.pipeline import answer_question

router = APIRouter(tags=["rag"])


def _distance_to_score(distance: float) -> float:
    # Chroma distances are "smaller is better".
    # Convert to a 0..1 score (simple, stable).
    # score = 1 / (1 + d)
    s = 1.0 / (1.0 + max(0.0, float(distance)))
    # clamp
    return max(0.0, min(1.0, s))


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    start = time.perf_counter()
    request_id = f"req_{uuid.uuid4().hex[:10]}"

    try:
        out = answer_question(
            question=req.question,
            top_k=req.top_k,
            doc_ids=req.doc_ids,     # list[str]
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
                score=_distance_to_score(dist),
            )
        )

    latency_ms = int((time.perf_counter() - start) * 1000)
    meta = Meta(
        request_id=request_id,
        latency_ms=latency_ms,
        model=settings.openai_chat_model if settings.model_provider == "openai" else settings.model_provider,
        prompt_version="ask_v1",
    )
    return AskResponse(answer=out["answer"], citations=citations, meta=meta)
