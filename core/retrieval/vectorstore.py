from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.retrieval.embedder import OpenAIEmbedder


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str,
        embedder: OpenAIEmbedder,
        collection_name: str = "guidelines",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.col = self.client.get_or_create_collection(name=collection_name)
        self.embedder = embedder

    def upsert_chunks(
        self,
        doc_id: str,
        title: str | None,
        source: str | None,
        category: str | None,
        chunks: List[Dict[str, Any]],  # [{"id","text","page"}]
    ) -> int:
        ids = [f"{doc_id}:{c['id']}" for c in chunks]
        docs = [c["text"] for c in chunks]
        metas = [
            {
                "doc_id": doc_id,
                "chunk_id": c["id"],
                "page": int(c["page"]),
                "title": title,
                "source": source,
                "category": category,
            }
            for c in chunks
        ]

        embs = self.embedder.embed(docs)
        self.col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        return len(ids)

    def query(
        self,
        question: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        q_emb = self.embedder.embed([question])[0]
        where = {"doc_id": doc_id} if doc_id else None

        res = self.col.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        out: List[Dict[str, Any]] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            out.append(
                {
                    "text": doc,
                    "meta": meta,
                    "distance": float(dist),
                }
            )
        return out
