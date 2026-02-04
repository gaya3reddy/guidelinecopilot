from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.config import settings
from apps.api.routers.health import router as health_router
from apps.api.routers.ingest import router as ingest_router
from apps.api.routers.ask import router as ask_router
from apps.api.routers.summarize import router as summarize_router


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version=settings.app_version)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten later
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(ask_router)
    app.include_router(summarize_router)

    return app


app = create_app()
