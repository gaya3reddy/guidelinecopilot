from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """Lowercase and split on non-alphanumeric chars.

    Keeps numbers intact so terms like "80%", "v/v", "formulation1" survive.
    We split on whitespace and punctuation except percent/slash/dot so that
    exact clinical terms ("80% v/v ethanol", "formulation I") are preserved
    as individual tokens.
    """
    text = text.lower()
    # keep alphanumeric, %, /, . — split on everything else
    tokens = re.findall(r"[a-z0-9%/\.]+", text)
    return tokens


class BM25Store:
    """In-memory BM25 index built from a flat list of chunk dicts.

    Each chunk dict must have:
        id   — unique chunk identifier (e.g. "doc_abc:chunk_0042")
        text — raw chunk text
        meta — metadata dict (doc_id, page, chunk_id, …)

    Usage
    -----
    store = BM25Store(chunks)          # build index
    results = store.search("query", top_k=10, doc_id="doc_abc")
    """

    def __init__(self, chunks: List[Dict[str, Any]]) -> None:
        self._chunks = chunks
        tokenized = [_tokenize(c["text"]) for c in chunks]
        # BM25Okapi handles empty corpus gracefully but we guard anyway
        self._index = BM25Okapi(tokenized) if chunks else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return up to top_k chunks ranked by BM25 score.

        Parameters
        ----------
        query:  the user question / search string
        top_k:  maximum results to return
        doc_id: if set, restrict results to this document
        """
        if self._index is None or not self._chunks:
            return []

        q_tokens = _tokenize(query)
        scores = self._index.get_scores(q_tokens)  # ndarray length = len(chunks)

        # Pair (score, chunk) and optionally filter by doc_id
        candidates = [
            (float(scores[i]), self._chunks[i])
            for i in range(len(self._chunks))
            if doc_id is None or self._chunks[i]["meta"].get("doc_id") == doc_id
        ]

        # Sort descending by score, take top_k
        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:top_k]

        return [
            {
                "text": chunk["text"],
                "meta": chunk["meta"],
                "bm25_score": score,
            }
            for score, chunk in top
        ]

    def rebuild(self, chunks: List[Dict[str, Any]]) -> None:
        """Replace the index with a fresh set of chunks (called after ingest)."""
        self.__init__(chunks)
