from __future__ import annotations

from typing import Any, Dict, List


def distance_to_score(distance: float) -> float:
    """Convert ChromaDB distance (lower = better) to a 0..1 similarity score. score = 1 / (1 + d)"""
    s = 1.0 / (1.0 + max(0.0, float(distance)))  # ensure distance is non-negative
    return max(0.0, min(1.0, s))  # clamp to [0,1]


def documents_to_citations(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert HybridRetriever.search() output into Citation-shaped dicts.

    Extracted from the loop that used to live inline in the /ask route,
    so both the plain and agentic ask endpoints build citations the
    same way. Returns plain dicts (not Citation objects) so this module
    doesn't need to import from core.schemas.models — callers wrap the
    result in Citation(**c) themselves.
    """
    citations = []
    for c in documents:
        meta = c["meta"]
        citations.append(
            {
                "doc_id": str(meta.get("doc_id", "")),
                "page": int(meta.get("page") or 0),
                "chunk_id": str(meta.get("chunk_id", "")),
                "snippet": c["text"],
                "score": distance_to_score(float(c["distance"])),
            }
        )
    return citations
