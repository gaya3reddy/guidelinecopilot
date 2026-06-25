from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.retrieval.bm25_store import BM25Store
from core.retrieval.vectorstore import ChromaVectorStore


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    id_key: str,
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Merge N ranked lists using Reciprocal Rank Fusion.

    RRF score for a document d = Σ  1 / (k + rank_in_list_i)
    Higher is better.  No score normalisation needed.

    Parameters
    ----------
    ranked_lists : each inner list is already ordered best-first.
    id_key       : key in each result dict used as the unique chunk ID.
    k            : smoothing constant (default 60 is standard).
    """
    scores: Dict[str, float] = {}
    items: Dict[str, Dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            uid = item[id_key]
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)
            items[uid] = item  # keep the most recently seen copy

    merged = sorted(items.values(), key=lambda x: scores[x[id_key]], reverse=True)
    # attach the fused score for transparency
    for item in merged:
        item["rrf_score"] = scores[item[id_key]]
    return merged


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Combines BM25 keyword search and dense vector search via RRF.

    Designed to be instantiated once at API startup and reused across
    requests.  Call `rebuild_bm25()` after every ingest so the keyword
    index stays fresh.

    Parameters
    ----------
    vector_store : ChromaVectorStore instance (already initialised)
    bm25_store   : BM25Store instance (already built from current chunks)
    bm25_candidates : how many BM25 results to fetch before RRF fusion
                      (should be >= top_k; more candidates → better recall)
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_store: BM25Store,
        bm25_candidates: int = 20,
    ) -> None:
        self._vs = vector_store
        self._bm25 = bm25_store
        self._bm25_candidates = bm25_candidates

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return top_k chunks fused from BM25 + vector search.

        Each result dict has the standard pipeline shape:
            text, meta, distance   (from vector leg)
        plus:
            bm25_score  (from BM25 leg, 0.0 if not in BM25 results)
            rrf_score   (final fused rank score — higher is better)
        """
        # --- BM25 leg ---
        bm25_results = self._bm25.search(
            query=query,
            top_k=self._bm25_candidates,
            doc_id=doc_id,
        )
        # Normalise BM25 results to share a common shape with vector results.
        # Use composite chunk id as the unique key.
        for r in bm25_results:
            r.setdefault("distance", 0.0)
            r["_uid"] = (
                r["meta"].get("doc_id", "") + ":" + r["meta"].get("chunk_id", "")
            )

        # --- Vector leg (fetch more than top_k for better fusion) ---
        vector_results = self._vs.query(
            question=query,
            top_k=self._bm25_candidates,
            doc_id=doc_id,
        )
        for r in vector_results:
            r["_uid"] = (
                r["meta"].get("doc_id", "") + ":" + r["meta"].get("chunk_id", "")
            )
            r.setdefault("bm25_score", 0.0)

        # --- Fuse via RRF ---
        fused = _reciprocal_rank_fusion(
            ranked_lists=[bm25_results, vector_results],
            id_key="_uid",
        )

        # Trim to top_k and clean up the internal key
        top = fused[:top_k]
        for item in top:
            item.pop("_uid", None)

        return top

    def rebuild_bm25(self) -> None:
        """Reload the BM25 index from ChromaDB.

        Call this at the end of every ingest so newly added chunks are
        immediately searchable via the keyword index.
        """
        all_chunks = self._vs.get_all_chunks()
        self._bm25.rebuild(all_chunks)
