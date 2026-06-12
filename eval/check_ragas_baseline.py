"""
eval/check_ragas_baseline.py
-----------------------------
CI gate — reads eval/baseline/ragas_baseline.json and fails if any
metric score is below its threshold. No API calls, no OpenAI, no cost.

Run manually:  python eval/check_ragas_baseline.py
Run in CI:     automatically called by .github/workflows/ci.yml
"""

import json
import sys
from pathlib import Path

BASELINE_PATH = Path(__file__).resolve().parent / "baseline" / "ragas_baseline.json"

THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.70,
    "context_precision": 0.60,
    "context_recall": 0.55,
}


def main() -> None:
    if not BASELINE_PATH.exists():
        print(f"[ERROR] Baseline not found at {BASELINE_PATH}")
        print("  Run: python -m eval.run_ragas_eval")
        sys.exit(1)

    baseline = json.loads(BASELINE_PATH.read_text())
    scores = baseline.get("scores", {})

    print("RAGAS Baseline Threshold Gate")
    print("=" * 50)

    all_passed = True
    for metric, threshold in THRESHOLDS.items():
        score = scores.get(metric)
        if score is None:
            print(f"  {metric:<22} MISSING in baseline")
            all_passed = False
            continue
        status = "PASS" if score >= threshold else "FAIL"
        if score < threshold:
            all_passed = False
        print(f"  {metric:<22} {score:.4f}  (min: {threshold})  {status}")

    print("=" * 50)
    if not all_passed:
        print("[FAIL] Baseline scores below threshold.")
        print("       Run python -m eval.run_ragas_eval to refresh baseline.")
        sys.exit(1)

    print("[PASS] All baseline scores meet thresholds.")


if __name__ == "__main__":
    main()
