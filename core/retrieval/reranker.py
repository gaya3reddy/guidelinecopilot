"""
Cross-encoder reranker for GuidelineCopilot.

Sits between HybridRetriever (RRF output) and the LLM prompt.
The cross-encoder reads (query, chunk_text) pairs jointly and scores
each pair's relevance directly — much more accurate than rank-position
fusion alone, especially for clinical terminology like "80% v/v ethanol".

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22M parameters, ~85 MB download
  - Trained on MS-MARCO passage retrieval
  - No GPU required; runs fine on CPU in Cloud Run
  - Outputs raw logit scores (higher = more relevant)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once per container, reused across requests
_cross_encoder = None


def _get_model():
    """Lazy-load and cache the CrossEncoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers.cross_encoder import CrossEncoder  # noqa: PLC0415

        from apps.api.config import settings  # noqa: PLC0415

        model_name = settings.reranker_model
        logger.info("Loading cross-encoder: %s", model_name)
        _cross_encoder = CrossEncoder(model_name)
        logger.info("Cross-encoder ready")
    return _cross_encoder


def rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Rerank *chunks* against *query* using a cross-encoder and return the
    top-*top_k* results sorted by relevance (highest score first).

    Args:
        query:  The user's question.
        chunks: List of chunk dicts with a ``"text"`` key.
        top_k:  How many chunks to return after reranking.

    Returns:
        Reranked slice of *chunks* (length ≤ top_k), best chunk first.
    """
    if not chunks:
        return chunks

    model = _get_model()

    pairs = [[query, chunk.get("text", "")] for chunk in chunks]
    scores: list[float] = model.predict(pairs, show_progress_bar=False).tolist()

    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    reranked = [chunk for _, chunk in scored[:top_k]]

    logger.info(
        "Reranker: %d → %d chunks | top score %.4f | bottom score %.4f",
        len(chunks),
        len(reranked),
        scored[0][0],
        scored[min(top_k, len(scored)) - 1][0],
    )
    return reranked
