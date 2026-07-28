"""
eval/run_ragas_eval_agentic.py
-------------------------------
RAGAS evaluation harness for GuidelineCopilot's /ask/agentic endpoint
(LangGraph self-correcting retrieval loop), for direct comparison against
the standard /ask pipeline's baseline scores.

Same golden dataset, same four RAGAS metrics, same methodology as
eval/run_ragas_eval.py — the only difference is which endpoint is called.
This intentionally does NOT write to eval/baseline/ragas_baseline.json
(the file CI's ragas-threshold-gate reads); see step 11 below.

  - faithfulness        (does the answer stay inside the retrieved context?)
  - answer_relevancy    (does the answer address the question?)
  - context_precision   (are retrieved chunks actually useful?)
  - context_recall      (did retrieval find everything needed?)

Usage:
    # API must be running first
    uvicorn apps.api.main:app --reload --port 8000

    # Run eval against /ask/agentic
    python -m eval.run_ragas_eval_agentic

    # Optional: point at a different host
    EVAL_BASE_URL=http://localhost:8000 python -m eval.run_ragas_eval_agentic

Output:
    - Summary table printed to console
    - eval/reports/ragas_report_agentic_YYYYMMDD_HHMMSS.json (full per-question scores)
    - eval/baseline/ragas_baseline_agentic.json              (overwritten each run — comparison only, NOT read by CI)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import numpy as np

# ---------------------------------------------------------------------------
# RAGAS imports
# ragas >= 0.1 uses a Dataset-based API.
# We import the four metrics and the evaluate() function.
# ---------------------------------------------------------------------------
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# ---------------------------------------------------------------------------
# Paths — all relative to project root so this works whether run locally
# or inside Docker (where CWD is /app).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden" / "who_hand_hygiene.json"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config — override with environment variables if needed
# ---------------------------------------------------------------------------
BASE_URL = os.getenv("EVAL_BASE_URL", "http://localhost:8000")
TOP_K = int(os.getenv("EVAL_TOP_K", "5"))  # same default as UI
TIMEOUT = int(os.getenv("EVAL_TIMEOUT_SEC", "60"))

# CI thresholds — if any metric falls below these, the script exits with
# code 1 so GitHub Actions marks the job as failed.
THRESHOLDS = {
    "faithfulness": 0.70,
    "answer_relevancy": 0.60,
    "context_precision": 0.60,
    "context_recall": 0.55,
}


# ---------------------------------------------------------------------------
# Helper: call POST /ask and return (answer, contexts_list)
# We use the non-streaming endpoint here for simplicity.
# If your API only exposes /ask/stream, swap this function out.
# ---------------------------------------------------------------------------
def call_ask(question: str, doc_id: str) -> tuple[str, list[str]]:
    """
    Returns:
        answer   - the generated answer string
        contexts - list of retrieved snippet strings (what RAGAS calls 'contexts')
    """
    payload = {
        "question": question,
        "doc_ids": [doc_id],
        "top_k": TOP_K,
        "mode": "rag",
    }

    resp = requests.post(
        f"{BASE_URL}/ask/agentic",
        json=payload,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    answer: str = data.get("answer", "")

    # citations[] is a list of dicts with a 'snippet' key.
    # RAGAS wants contexts as plain strings — extract the snippet text.
    citations: list[dict] = data.get("citations", [])
    contexts: list[str] = [c["snippet"] for c in citations if c.get("snippet")]
    # contexts: list[str] = [c["text"] for c in citations if c.get("text")]

    # Guard: RAGAS will error if contexts is empty.
    # For unanswerable questions the model may still return an answer
    # but retrieval might return empty. Use a sentinel so RAGAS can score.
    if not contexts:
        contexts = ["[no context retrieved]"]

    return answer, contexts


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_eval() -> dict:
    # 1. Load the golden dataset
    if not GOLDEN_PATH.exists():
        print(f"[ERROR] Golden dataset not found at {GOLDEN_PATH}")
        print("  Run: cp /path/to/who_hand_hygiene.json eval/golden/")
        sys.exit(1)

    golden: list[dict] = json.loads(GOLDEN_PATH.read_text())
    print(f"\n{'=' * 60}")
    print("GuidelineCopilot — RAGAS Evaluation")
    print(f"{'=' * 60}")
    print(f"Golden dataset : {GOLDEN_PATH.name}  ({len(golden)} questions)")
    print(f"API endpoint   : {BASE_URL}")
    print(f"top_k          : {TOP_K}")
    print(f"{'=' * 60}\n")

    # 2. Check API is reachable before starting
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        health.raise_for_status()
    except Exception as e:
        print(f"[ERROR] API not reachable at {BASE_URL}/health — {e}")
        print("  Start it with: uvicorn apps.api.main:app --reload --port 8000")
        sys.exit(1)

    # 3. Collect answers and contexts for every question
    # RAGAS expects four parallel lists that map to each other by index.
    questions: list[str] = []
    answers: list[str] = []
    contexts_list: list[list[str]] = []
    ground_truths: list[str] = []
    question_types: list[str] = []  # stored for per-type breakdown, not passed to RAGAS

    print("Calling /ask for each question...")
    for i, row in enumerate(golden):
        q = row["question"]
        gt = row["ground_truth"]
        doc_id = row["doc_id"]
        qtype = row.get("type", "unknown")

        print(f"  [{i + 1:02d}/{len(golden)}] ({qtype:>16})  {q[:60]}...")

        try:
            answer, contexts = call_ask(q, doc_id)
        except requests.HTTPError as e:
            print(f"    [WARN] HTTP error for question {i + 1}: {e} — skipping")
            continue
        except Exception as e:
            print(f"    [WARN] Unexpected error for question {i + 1}: {e} — skipping")
            continue

        questions.append(q)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(gt)
        question_types.append(qtype)

        # Small delay to avoid hitting OpenAI rate limits during RAGAS scoring
        time.sleep(0.3)

    if not questions:
        print("[ERROR] No questions were evaluated successfully.")
        sys.exit(1)

    print(f"\nSuccessfully collected {len(questions)}/{len(golden)} responses.\n")

    # 4. Build a HuggingFace Dataset — this is what ragas.evaluate() expects
    #    Column names must match exactly: question, answer, contexts, ground_truth
    ragas_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
    )

    # 5. Run RAGAS scoring
    # RAGAS uses an LLM judge internally — it will use OPENAI_API_KEY from env.
    # gpt-4o-mini is the default and is cheap (~$0.02 for 20 questions).
    print("Running RAGAS scoring (this calls the OpenAI API ~4x per question)...")
    print("Expected cost: ~$0.02–0.05 for 20 questions with gpt-4o-mini\n")

    result = evaluate(
        dataset=ragas_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    # # 6. Extract scores
    # scores: dict[str, float] = {
    #     "faithfulness": round(float(result["faithfulness"]), 4),
    #     "answer_relevancy": round(float(result["answer_relevancy"]), 4),
    #     "context_precision": round(float(result["context_precision"]), 4),
    #     "context_recall": round(float(result["context_recall"]), 4),
    # }
    # 6. Extract scores
    # RAGAS 0.2.x returns per-question lists — take the mean
    def _mean(vals) -> float:
        v = [
            x
            for x in vals
            if x is not None and not (isinstance(x, float) and np.isnan(x))
        ]
        return round(float(np.mean(v)), 4) if v else 0.0

    scores: dict[str, float] = {
        "faithfulness": _mean(result["faithfulness"]),
        "answer_relevancy": _mean(result["answer_relevancy"]),
        "context_precision": _mean(result["context_precision"]),
        "context_recall": _mean(result["context_recall"]),
    }

    # 7. Per-question detail (result.to_pandas() gives row-level scores)
    df = result.to_pandas()
    df["question_type"] = question_types[: len(df)]
    df["question_type"] = question_types[: len(df)]

    per_question: list[dict] = []
    for _, row in df.iterrows():
        per_question.append(
            {
                "question": row["user_input"],
                "question_type": row.get("question_type", "unknown"),
                "answer": row["response"],
                "ground_truth": row["reference"],
                "faithfulness": round(float(row.get("faithfulness") or 0), 4),
                "answer_relevancy": round(float(row.get("answer_relevancy") or 0), 4),
                "context_precision": round(float(row.get("context_precision") or 0), 4),
                "context_recall": round(float(row.get("context_recall") or 0), 4),
            }
        )

    # 8. Per-type breakdown — useful for debugging which question types struggle
    type_breakdown: dict[str, dict] = {}
    for qtype in set(question_types):
        rows = [r for r in per_question if r["question_type"] == qtype]
        type_breakdown[qtype] = {
            metric: round(sum(r[metric] for r in rows) / len(rows), 4)
            for metric in scores
        }

    # 9. Print summary to console
    print(f"\n{'=' * 60}")
    print("RAGAS Results — Summary")
    print(f"{'=' * 60}")
    all_passed = True
    for metric, score in scores.items():
        threshold = THRESHOLDS[metric]
        status = "✓ PASS" if score >= threshold else "✗ FAIL"
        if score < threshold:
            all_passed = False
        print(f"  {metric:<22} {score:.4f}   (threshold: {threshold})  {status}")

    print(f"\n{'=' * 60}")
    print("Per question-type breakdown")
    print(f"{'=' * 60}")
    for qtype, type_scores in type_breakdown.items():
        print(f"  {qtype}:")
        for metric, score in type_scores.items():
            print(f"    {metric:<22} {score:.4f}")

    # 10. Save full timestamped report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": timestamp,
        "golden_dataset": GOLDEN_PATH.name,
        "num_questions": len(questions),
        "top_k": TOP_K,
        "scores": scores,
        "thresholds": THRESHOLDS,
        "all_passed": all_passed,
        "type_breakdown": type_breakdown,
        "per_question": per_question,
    }

    timestamped_path = REPORTS_DIR / f"ragas_report_agentic_{timestamp}.json"
    timestamped_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Full report saved → {timestamped_path}")

    # 11. NOTE: deliberately NOT writing to eval/baseline/ragas_baseline.json.
    # That file is read by CI's ragas-threshold-gate and represents the
    # standard /ask pipeline. Overwriting it here would silently change
    # what CI is gating on. This comparison run gets its own file instead.
    comparison_path = PROJECT_ROOT / "eval" / "baseline" / "ragas_baseline_agentic.json"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(report, indent=2))
    print(f"  Agentic comparison saved → {comparison_path}")
    print("  (eval/baseline/ragas_baseline.json — the CI baseline — was NOT touched)")

    # 12. Exit with error code if any threshold was breached (for CI)
    if not all_passed:
        print("\n[FAIL] One or more metrics are below threshold — see above.")
        sys.exit(1)

    print("\n[PASS] All metrics above threshold.")
    return report


if __name__ == "__main__":
    run_eval()
