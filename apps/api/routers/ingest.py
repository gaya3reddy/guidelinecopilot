from __future__ import annotations

import time
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from core.schemas.models import IngestResponse, DocInfo, DocList
from apps.api.config import settings
import hashlib

router = APIRouter(tags=["ingestion"])

# Temporary in-memory doc registry for Day-1.
# Day-2 will replace with persistent metadata (JSON) in data/processed/.
_DOCS: dict[str, DocInfo] = {}
_HASH_TO_DOC: dict[str, str] = {}


@router.get("/documents", response_model=DocList)
def list_docs() -> DocList:
    return DocList(items=list(_DOCS.values()))


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    doc_id: str | None = Form(default=None),
    title: str | None = Form(default=None),
    source: str | None = Form(default=None),
    category: str | None = Form(default=None),
) -> IngestResponse:
    start = time.perf_counter()
    if file.content_type not in ("application/pdf",):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # File size guard (best-effort - relies on reading)
    data = await file.read()
    file_hash = hashlib.sha256(data).hexdigest()

    # If same PDF already ingested, return existing doc_id
    if file_hash in _HASH_TO_DOC:
        existing_id = _HASH_TO_DOC[file_hash]
        return IngestResponse(
            doc_id=existing_id,
            chunks_indexed=0,
            pages=0,
            deduped=True,
            message="Duplicate PDF detected — returning existing doc_id."
        )

    
    
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Max {settings.max_upload_mb} MB.")

        
    # Choose doc_id
    # safe_doc_id = (doc_id or f"doc_{uuid.uuid4().hex[:8]}").strip()
    safe_doc_id = (doc_id or "").strip()
    if not safe_doc_id or safe_doc_id.lower() == "string":
        safe_doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    
    _HASH_TO_DOC[file_hash] = safe_doc_id
    
    if safe_doc_id in _DOCS:
        raise HTTPException(status_code=400, detail=f"doc_id '{safe_doc_id}' already exists. Choose a different doc_id.")

    # Save raw PDF (Day-2 will also extract pages)
    out_path = settings.raw_dir / f"{safe_doc_id}.pdf"
    out_path.write_bytes(data)

    # Register doc metadata
    _DOCS[safe_doc_id] = DocInfo(doc_id=safe_doc_id, title=title, source=source, category=category)

    # Stub counts for Day-1
    latency_ms = int((time.perf_counter() - start) * 1000)
    _ = latency_ms  # placeholder

    return IngestResponse(doc_id=safe_doc_id, chunks_indexed=0, pages=0)
