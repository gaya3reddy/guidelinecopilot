from __future__ import annotations


def distance_to_score(distance: float) -> float:
    """Convert ChromaDB distance (lower = better) to a 0..1 similarity score. score = 1 / (1 + d)"""
    s = 1.0 / (1.0 + max(0.0, float(distance)))  # ensure distance is non-negative
    return max(0.0, min(1.0, s))  # clamp to [0,1]
