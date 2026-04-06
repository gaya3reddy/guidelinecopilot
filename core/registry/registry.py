from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from core.schemas.models import DocInfo

_lock = Lock()


class DocumentRegistry:
    def __init__(self, registry_path: Path):
        self.path = registry_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]")

    def _read(self) -> list[dict]:
        with _lock:
            return json.loads(self.path.read_text())

    def _write(self, docs: list[dict]) -> None:
        with _lock:
            self.path.write_text(json.dumps(docs, indent=2))

    def all(self) -> list[DocInfo]:
        return [DocInfo(**d) for d in self._read()]

    def get(self, doc_id: str) -> DocInfo | None:
        for d in self._read():
            if d["doc_id"] == doc_id:
                return DocInfo(**d)
        return None

    def exists(self, doc_id: str) -> bool:
        return self.get(doc_id) is not None

    def get_by_hash(self, file_hash: str) -> str | None:
        for d in self._read():
            if d.get("file_hash") == file_hash:
                return d["doc_id"]
        return None

    def add(self, doc: DocInfo, file_hash: str) -> None:
        docs = self._read()
        docs.append({**doc.model_dump(), "file_hash": file_hash})
        self._write(docs)

    def delete(self, doc_id: str) -> bool:
        docs = self._read()
        new_docs = [d for d in docs if d["doc_id"] != doc_id]
        if len(new_docs) == len(docs):
            return False
        self._write(new_docs)
        return True
