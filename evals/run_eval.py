#!/usr/bin/env python3
"""Run the held-out set against a running instance.

    python evals/run_eval.py --url http://localhost:8000 --out evals/results.json

Exits non-zero if any gate regresses, so CI can block a prompt change that
quietly makes the thing answer questions it should refuse.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from evals.metrics import CaseResult, summarise

GATES = {"refusal_accuracy": 1.0, "citation_precision": 0.8, "mean_grounding": 0.7}
REFUSAL_MARKERS = ("do not cover", "does not contain", "not in the indexed")


async def run_case(client: httpx.AsyncClient, case: dict) -> CaseResult:
    started = time.perf_counter()
    resp = await client.post("/ask", json={"question": case["question"], "use_cache": False})
    resp.raise_for_status()
    body = resp.json()
    elapsed = (time.perf_counter() - started) * 1000

    answer = body.get("answer", "")
    refused = any(m in answer.lower() for m in REFUSAL_MARKERS) or not body.get("citations")
    cites = " ".join(body.get("citations", []))

    return CaseResult(
        id=case["id"],
        answered=not refused,
        expected_refusal=case.get("expect_refusal", False),
        citation_hit=any(s in cites for s in case.get("expected_sources", [])),
        contains_ok=all(t in answer for t in case.get("must_contain", [])),
        forbidden_ok=not any(t in answer for t in case.get("must_not_contain", [])),
        retrieval_score=body.get("retrieval_score") or 0.0,
        grounding_score=body.get("grounding_score") or 0.0,
        latency_ms=elapsed,
    )


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--dataset", default="evals/dataset.jsonl")
    ap.add_argument("--out", default="evals/results.json")
    args = ap.parse_args()

    cases = [json.loads(l) for l in Path(args.dataset).read_text().splitlines() if l.strip()]

    async with httpx.AsyncClient(base_url=args.url, timeout=120) as client:
        results = await asyncio.gather(*(run_case(client, c) for c in cases))

    report = summarise(list(results))
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    failures = [f"{k}: {report[k]} < {v}" for k, v in GATES.items() if report[k] < v]
    for f in failures:
        print(f"GATE FAILED  {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
