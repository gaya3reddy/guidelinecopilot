from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from eval.metrics import summarize_metrics


DATASET_PATH = Path("eval/dataset.jsonl")
REPORTS_DIR = Path("eval/reports")


import os

BASE_URL = os.getenv("EVAL_BASE_URL", "http://localhost:8000").rstrip("/")

def endpoint_for(kind: str) -> str:
    if kind == "ask":
        return f"{BASE_URL}/ask"
    if kind == "summarize":
        return f"{BASE_URL}/summarize"
    raise ValueError(f"Unknown type: {kind}")



def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("//") or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def post_json(url: str, payload: dict):
    r = requests.post(url, json=payload, timeout=60)
    if r.status_code == 422:
        print("\n--- 422 VALIDATION ERROR ---")
        print("Payload:", payload)
        print("Server says:", r.text)
        print("----------------------------\n")
    r.raise_for_status()
    return r.json()

def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_jsonl(DATASET_PATH)
    results: List[Dict[str, Any]] = []

    for i, ex in enumerate(dataset, start=1):
        kind = ex.get("type", "ask")  # "ask" or "summarize"
        
        payload = dict(ex)
        payload.pop("type", None)
        payload.pop("id", None)

        # payload = ex.get("payload") or {}

        url = endpoint_for(kind)

        t0 = time.perf_counter()
        resp = post_json(url, payload)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        
             
        results.append(
            {
                "id": ex.get("id", f"ex_{i}"),
                "endpoint": kind,          # <-- IMPORTANT: "ask" or "summarize"
                "url": url,                # optional, but useful for debugging
                "mode": payload.get("mode"),
                "latency_ms": dt_ms,
                "request": payload,
                "response": resp,
            }
        )


        print(f"[{i}/{len(dataset)}] {kind} {dt_ms:.0f} ms")

    metrics = summarize_metrics(results)

    out = {
        "base_url": BASE_URL,
        "n": len(results),
        "metrics": metrics,
        "samples": results,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"report_{ts}.json"
    report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== METRICS ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
