from __future__ import annotations

import time
import uuid
from fastapi import APIRouter
from core.schemas.models import SummarizeRequest, SummarizeResponse, Meta

router = APIRouter(tags=["summarize"])


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest) -> SummarizeResponse:
    start = time.perf_counter()
    request_id = f"req_{uuid.uuid4().hex[:10]}"

    summary = (
        f"Summarization not implemented yet (Day-1 skeleton). "
        f"Requested style: {req.style}. Next: retrieve key chunks -> generate grounded summary with citations."
    )

    latency_ms = int((time.perf_counter() - start) * 1000)
    meta = Meta(
        request_id=request_id,
        latency_ms=latency_ms,
        model="stub",
        prompt_version="summarize_v1",
    )
    return SummarizeResponse(summary=summary, citations=[], meta=meta)
