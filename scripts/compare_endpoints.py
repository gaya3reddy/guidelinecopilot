"""
Compares /ask vs /ask/agentic head-to-head on latency and retry behavior,
using the same golden dataset the RAGAS eval runs use — so the cost side
of the tradeoff can be read next to the quality numbers from
eval/run_ragas_eval.py and eval/run_ragas_eval_agentic.py.

This does NOT run RAGAS scoring (no OpenAI judge calls) — it only measures
what each endpoint's own Meta.latency_ms reports, which is already
instrumented server-side in apps/api/routers/ask.py. Cheap and fast to
re-run any time you want an updated latency comparison.

Usage:
    # API must be running first
    uvicorn apps.api.main:app --reload --port 8000

    python -m scripts.compare_endpoints
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden" / "who_hand_hygiene.json"
BASE_URL = os.getenv("EVAL_BASE_URL", "http://localhost:8000")
TIMEOUT = int(os.getenv("EVAL_TIMEOUT_SEC", "60"))


def call(endpoint: str, question: str, doc_id: str) -> dict:
    resp = requests.post(
        f"{BASE_URL}{endpoint}",
        json={"question": question, "doc_ids": [doc_id], "top_k": 5, "mode": "rag"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    if not GOLDEN_PATH.exists():
        print(f"[ERROR] Golden dataset not found at {GOLDEN_PATH}")
        sys.exit(1)

    golden: list[dict] = json.loads(GOLDEN_PATH.read_text())

    try:
        requests.get(f"{BASE_URL}/health", timeout=5).raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] API not reachable at {BASE_URL}/health — {e}")
        print("  Start it with: uvicorn apps.api.main:app --reload --port 8000")
        sys.exit(1)

    rows = []
    print(f"Comparing /ask vs /ask/agentic across {len(golden)} questions...\n")

    for i, item in enumerate(golden):
        q, doc_id, qtype = item["question"], item["doc_id"], item.get("type", "unknown")
        print(f"  [{i + 1:02d}/{len(golden)}] ({qtype:>16})  {q[:55]}...")

        try:
            plain = call("/ask", q, doc_id)
            agentic = call("/ask/agentic", q, doc_id)
        except Exception as e:  # noqa: BLE001 — deliberately broad: one bad
            # question must not kill the rest of a 20-question comparison run
            print(f"    [WARN] skipped — {e}")
            continue

        rows.append(
            {
                "question_type": qtype,
                "plain_ms": plain["meta"]["latency_ms"],
                "agentic_ms": agentic["meta"]["latency_ms"],
                "retries": agentic["meta"].get("retries", 0),
            }
        )

    if not rows:
        print("[ERROR] No comparisons succeeded.")
        sys.exit(1)

    n = len(rows)
    avg_plain = sum(r["plain_ms"] for r in rows) / n
    avg_agentic = sum(r["agentic_ms"] for r in rows) / n
    total_retries = sum(r["retries"] for r in rows)
    questions_with_retry = sum(1 for r in rows if r["retries"] > 0)

    print(f"\n{'=' * 60}")
    print("Latency Comparison — /ask vs /ask/agentic")
    print(f"{'=' * 60}")
    print(f"  Questions compared     : {n}")
    print(f"  Avg latency /ask       : {avg_plain:,.0f} ms")
    print(f"  Avg latency /ask/agentic: {avg_agentic:,.0f} ms")
    print(
        f"  Overhead                : {avg_agentic - avg_plain:,.0f} ms "
        f"({(avg_agentic / avg_plain - 1) * 100:+.1f}%)"
    )
    print(f"  Questions that retried  : {questions_with_retry}/{n}")
    print(f"  Total retries fired     : {total_retries}")

    # Per-type breakdown — this is the interesting part: overhead should
    # concentrate on the question types that actually need retries
    # (unanswerable, maybe synthesis), not spread evenly across all types.
    print(f"\n{'=' * 60}")
    print("Per question-type breakdown")
    print(f"{'=' * 60}")
    types = sorted({r["question_type"] for r in rows})
    for qtype in types:
        type_rows = [r for r in rows if r["question_type"] == qtype]
        tn = len(type_rows)
        t_plain = sum(r["plain_ms"] for r in type_rows) / tn
        t_agentic = sum(r["agentic_ms"] for r in type_rows) / tn
        t_retries = sum(r["retries"] for r in type_rows)
        print(
            f"  {qtype:<16} avg /ask: {t_plain:>7,.0f} ms   "
            f"avg /ask/agentic: {t_agentic:>7,.0f} ms   "
            f"retries: {t_retries}"
        )

    out_path = PROJECT_ROOT / "eval" / "reports" / "latency_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\n  Full per-question data saved → {out_path}")


if __name__ == "__main__":
    main()
