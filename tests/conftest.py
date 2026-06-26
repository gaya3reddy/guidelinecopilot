"""
Shared pytest fixtures for GuidelineCopilot tests.
No external services (OpenAI / ChromaDB) are touched here.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Citation fixtures — represent what ChromaVectorStore.query() returns
# ---------------------------------------------------------------------------


@pytest.fixture
def single_citation() -> list[dict]:
    return [
        {
            "text": "Hand hygiene is the single most important measure to prevent infection.",
            "meta": {"doc_id": "doc_abc123", "page": 3, "chunk_id": "p3_c0"},
            "distance": 0.12,
        }
    ]


@pytest.fixture
def multi_citations() -> list[dict]:
    return [
        {
            "text": "Wash hands with soap and water for at least 20 seconds.",
            "meta": {"doc_id": "doc_abc123", "page": 1, "chunk_id": "p1_c0"},
            "distance": 0.10,
        },
        {
            "text": "Alcohol-based hand rubs are effective against most pathogens.",
            "meta": {"doc_id": "doc_abc123", "page": 2, "chunk_id": "p2_c0"},
            "distance": 0.18,
        },
        {
            "text": "Use PPE when in contact with blood or bodily fluids.",
            "meta": {"doc_id": "doc_xyz999", "page": 5, "chunk_id": "p5_c1"},
            "distance": 0.25,
        },
    ]
