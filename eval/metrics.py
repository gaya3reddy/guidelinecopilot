from __future__ import annotations

from typing import Any, Dict, List
import re
import statistics


def safe_mean(xs: List[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def percentile(xs: List[float], p: float) -> float:
    """p in [0, 100]"""
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs_sorted) - 1)
    if f == c:
        return float(xs_sorted[f])
    return float(xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f))


def has_citations(resp: Dict[str, Any]) -> bool:
    cits = resp.get("citations") or []
    return len(cits) > 0


def answer_mentions_docs(answer: str) -> List[str]:
    """
    Your UI shows citations like: (doc_xxxx, p.2) sometimes.
    Extract doc_ ids if present.
    """
    return re.findall(r"(doc_[a-zA-Z0-9]+)", answer or "")


def grounding_overlap_score(answer: str, citations: List[Dict[str, Any]]) -> float:
    """
    Cheap faithfulness heuristic:
    - concatenate retrieved snippets
    - compute token overlap ratio with answer keywords (very rough)
    """
    if not answer or not citations:
        return 0.0

    evidence_text = " ".join([(c.get("snippet") or "") for c in citations]).lower()
    # keep only simple keywords
    answer_terms = re.findall(r"[a-zA-Z]{4,}", answer.lower())
    if not answer_terms:
        return 0.0
    answer_terms = list(dict.fromkeys(answer_terms))  # unique preserve order

    hits = sum(1 for t in answer_terms if t in evidence_text)
    return hits / max(1, len(answer_terms))


def summarize_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    rows: list of per-sample outputs created by run_eval
    expected keys per row:
      - endpoint: "ask" or "summarize"
      - latency_ms: float
      - response: dict (API json)
      - mode: optional ("rag"/"no_rag") for ask
    """
    out: Dict[str, Any] = {}

    for endpoint in ["ask", "summarize"]:
        subset = [r for r in rows if r.get("endpoint") == endpoint]
        lat = [float(r.get("latency_ms", 0.0)) for r in subset]

        out[endpoint] = {
            "count": len(subset),
            "latency_ms": {
                "avg": safe_mean(lat),
                "p95": percentile(lat, 95),
                "max": max(lat) if lat else 0.0,
            },
        }

        # citation coverage + grounding only for responses that have citations
        if endpoint == "ask":
            rag_subset = [r for r in subset if (r.get("mode") or "rag") == "rag"]
            cov = [
                1.0 if has_citations(r.get("response", {})) else 0.0 for r in rag_subset
            ]

            grounding_scores = []
            for r in rag_subset:
                resp = r.get("response") or {}
                ans = resp.get("answer") or ""
                cits = resp.get("citations") or []
                grounding_scores.append(grounding_overlap_score(ans, cits))

            out[endpoint]["citation_coverage_rag"] = safe_mean(cov)
            out[endpoint]["grounding_overlap_rag_avg"] = safe_mean(grounding_scores)

        if endpoint == "summarize":
            cov = [1.0 if has_citations(r.get("response", {})) else 0.0 for r in subset]
            grounding_scores = []
            for r in subset:
                resp = r.get("response") or {}
                summ = resp.get("summary") or ""
                cits = resp.get("citations") or []
                grounding_scores.append(grounding_overlap_score(summ, cits))

            out[endpoint]["citation_coverage"] = safe_mean(cov)
            out[endpoint]["grounding_overlap_avg"] = safe_mean(grounding_scores)

    return out
