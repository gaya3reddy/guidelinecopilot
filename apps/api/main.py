from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from apps.api.config import settings
from apps.api.routers.health import router as health_router
from apps.api.routers.ingest import router as ingest_router
from apps.api.routers.ask import router as ask_router
from apps.api.routers.summarize import router as summarize_router
from core.logging_config import get_logger

logger = get_logger("api.middleware")


def configure_uvicorn_logging() -> None:
    """
    Route uvicorn's loggers through our JSON formatter so all logs
    go to stdout as structured JSON on Cloud Run instead of plain
    text to stderr.
    """
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        # Remove uvicorn's default stderr handlers
        uv_logger.handlers.clear()
        # Re-attach using our formatter (stdout + JSON on GCP)
        get_logger(name)
        uv_logger.propagate = False


# Configure at import time so it takes effect before the server starts
configure_uvicorn_logging()


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version=settings.app_version)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten later
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start = time.perf_counter()

        response = await call_next(request)

        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"{request.method} {request.url.path} → {response.status_code}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        return response

    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(ask_router)
    app.include_router(summarize_router)

    return app


app = create_app()
