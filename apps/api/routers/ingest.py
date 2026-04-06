from __future__ import annotations

import hashlib
import time
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from apps.api.config import settings
from core.ingestion.chunker import chunk_pages
from core.ingestion.pdf_loader import extract_pages
from core.registry.registry import DocumentRegistry
from core.retrieval.embedder import OpenAIEmbedder
from core.retrieval.vectorstore import ChromaVectorStore
from core.schemas.models import DocInfo, DocList, IngestResponse

router = APIRouter(tags=["ingestion"])

registry = DocumentRegistry(settings.processed_dir / "registry.json")


@router.get("/documents", response_model=DocList)
def list_docs() -> DocList:
    return DocList(items=registry.all())


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

    data = await file.read()
    file_hash = hashlib.sha256(data).hexdigest()

    # Deduplication — check by hash
    existing_id = registry.get_by_hash(file_hash)
    if existing_id:
        return IngestResponse(
            doc_id=existing_id,
            chunks_indexed=0,
            pages=0,
            deduped=True,
            message="Duplicate PDF detected — returning existing doc_id.",
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {settings.max_upload_mb} MB.",
        )

    safe_doc_id = (doc_id or "").strip()
    if not safe_doc_id or safe_doc_id.lower() == "string":
        safe_doc_id = f"doc_{uuid.uuid4().hex[:8]}"

    if registry.exists(safe_doc_id):
        raise HTTPException(
            status_code=400,
            detail=f"doc_id '{safe_doc_id}' already exists.",
        )

    # Save raw PDF
    out_path = settings.raw_dir / f"{safe_doc_id}.pdf"
    out_path.write_bytes(data)

    # Register metadata
    doc_info = DocInfo(
        doc_id=safe_doc_id,
        title=title,
        source=source,
        category=category,
    )
    registry.add(doc_info, file_hash)

    # Extract, chunk, embed
    pages = extract_pages(data)
    page_pairs = [(p.page, p.text) for p in pages]
    chunks = chunk_pages(page_pairs)

    if settings.model_provider != "openai":
        raise HTTPException(
            status_code=400,
            detail="Only openai provider supported.",
        )
    if not settings.openai_api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is missing.")

    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key, model=settings.openai_embed_model
    )
    store = ChromaVectorStore(
        persist_dir=str(settings.processed_dir / "chroma"), embedder=embedder
    )

    chunks_indexed = store.upsert_chunks(
        doc_id=safe_doc_id,
        title=title,
        source=source,
        category=category,
        chunks=[{"id": c.chunk_id, "text": c.text, "page": c.page} for c in chunks],
    )

    latency_ms = int((time.perf_counter() - start) * 1000)
    _ = latency_ms

    return IngestResponse(
        doc_id=safe_doc_id,
        chunks_indexed=chunks_indexed,
        pages=len(pages),
        deduped=False,
        message=None,
    )
