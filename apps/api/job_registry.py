from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


@dataclass
class IngestJob:
    job_id: str
    doc_id: str
    status: JobStatus = JobStatus.PENDING
    progress: int = 0  # 0-100
    total_chunks: int = 0
    indexed_chunks: int = 0
    pages: int = 0
    error: Optional[str] = None
    message: Optional[str] = None


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, IngestJob] = {}
        self._lock = Lock()

    def create(self, job_id: str, doc_id: str) -> IngestJob:
        job = IngestJob(job_id=job_id, doc_id=doc_id)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> IngestJob | None:
        return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in kwargs.items():
                    setattr(job, k, v)
                # auto-calculate progress
                if job.total_chunks > 0:
                    job.progress = int((job.indexed_chunks / job.total_chunks) * 100)

    def set_done(self, job_id: str, pages: int, chunks: int) -> None:
        self.update(
            job_id,
            status=JobStatus.DONE,
            progress=100,
            pages=pages,
            indexed_chunks=chunks,
            total_chunks=chunks,
        )

    def set_error(self, job_id: str, error: str) -> None:
        self.update(
            job_id,
            status=JobStatus.ERROR,
            error=error,
        )


# Singleton — shared across the whole API process
job_registry = JobRegistry()
