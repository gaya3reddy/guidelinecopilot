from __future__ import annotations

import time
import uuid
from fastapi import APIRouter
from core.schemas.models import AskRequest, AskResponse, Meta

router = APIRouter(tags=["rag"])


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    start = time.perf_counter()
    request_id = f"req_{uuid.uuid4().hex[:10]}"

    # Day-1 stub response
    answer = (
        "RAG pipeline not implemented yet (Day-1 skeleton). "
        "Next: ingestion -> chunking -> embeddings -> vector search -> grounded answer with citations."
    )

    latency_ms = int((time.perf_counter() - start) * 1000)
    meta = Meta(
        request_id=request_id,
        latency_ms=latency_ms,
        model="stub",
        prompt_version="ask_v1",
    )
    return AskResponse(answer=answer, citations=[], meta=meta)
