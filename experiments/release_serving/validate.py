"""Validate a running release server against the recorded optimized rows.

Scores a target's frozen ID and OOD workloads the way the recorded campaign
did (workload order, sequential batches of 128 rows, a fresh client per
batch, so batch composition matches) and compares
them with the rows of the run whose scores the model card reports: per-row
score agreement, source-macro pAUROC@20 on both, and prompt throughput.
Labels come from the 4B records (both models share the row IDs).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.release_serving.targets import ROOT, TARGETS
from gleipnir.data.monitoring import read_rows, write_json, write_rows
from gleipnir.evaluation.http_score import score_batch
from gleipnir.evaluation.metrics import evaluate_binary_monitor

GATES = {"score_mae": 0.005, "correlation": 0.999, "pauroc_difference": 0.005}
ID_SOURCES = {"gloom_exfiltration", "test_stride"}
BATCH_ROWS = 128


def rows(path: str) -> list[dict]:
    full = ROOT / path
    return json.loads(full.read_text()) if full.suffix == ".json" else read_rows(full)


def labels(split: str) -> dict[str, tuple[str, int]]:
    if split == "id":
        workload = rows(TARGETS["4b"].validation["id"][0])
        return {r["id"]: (r["dataset"], int(r["label"])) for r in workload}
    recorded = read_rows(ROOT / TARGETS["4b"].validation["ood"][1])
    return {
        r["id"]: (r["original_metadata"]["source_dataset"], int(r["ground_truth"]))
        for r in recorded
    }


def macro_pauroc(scores: dict[str, float], truth: dict, split: str) -> float:
    frame = pd.DataFrame(
        [
            {"source": truth[i][0], "label": truth[i][1], "score": s}
            for i, s in scores.items()
        ]
    )
    if split == "id":
        frame = frame.loc[frame["source"].isin(ID_SOURCES)]
    return evaluate_binary_monitor(frame, group_column="source")["macro"][
        "pauroc_at_20"
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(TARGETS))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--splits", nargs="+", choices=("id", "ood"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rows-output", type=Path, help="directory for per-split scored rows"
    )
    args = parser.parse_args()
    target = TARGETS[args.model]
    config = {"port": args.port, "concurrency": 128, "timeout_seconds": 600}
    report = {"model": args.model, "gates": GATES, "splits": {}}
    for split, (workload_path, recorded_path, card) in target.validation.items():
        if args.splits and split not in args.splits:
            continue
        workload = rows(workload_path)
        recorded = {r["id"]: float(r["score"]) for r in read_rows(ROOT / recorded_path)}
        if {r["id"] for r in workload} != set(recorded):
            raise ValueError(f"{split} workload and recorded rows differ")
        scored, seconds = [], 0.0
        for start in range(0, len(workload), BATCH_ROWS):
            values, elapsed = asyncio.run(
                score_batch(workload[start : start + BATCH_ROWS], config)
            )
            scored += values
            seconds += elapsed
        new = {r["id"]: float(r["score"]) for r in scored}
        if args.rows_output:
            args.rows_output.mkdir(parents=True, exist_ok=True)
            write_rows(
                args.rows_output / f"{split}.jsonl",
                [
                    {"id": r["id"], "score": r["score"], "margin": r["margin"]}
                    for r in scored
                ],
            )
        ids = sorted(new)
        a = np.array([new[i] for i in ids])
        b = np.array([recorded[i] for i in ids])
        truth = labels(split)
        tokens = sum(r["prompt_tokens"] for r in workload)
        result = {
            "rows": len(ids),
            "score_mae": float(np.abs(a - b).mean()),
            "score_max_abs": float(np.abs(a - b).max()),
            "correlation": float(np.corrcoef(a, b)[0, 1]),
            "pauroc_at_20": macro_pauroc(new, truth, split),
            "recorded_pauroc_at_20": macro_pauroc(recorded, truth, split),
            "card_pauroc_at_20": card,
            "seconds": seconds,
            "prompt_tokens": tokens,
            "prompt_tokens_per_second": tokens / seconds,
            "median_latency_seconds": statistics.median(
                r["latency_seconds"] for r in scored
            ),
        }
        result["passed"] = (
            result["score_mae"] <= GATES["score_mae"]
            and result["correlation"] >= GATES["correlation"]
            and abs(result["pauroc_at_20"] - result["recorded_pauroc_at_20"])
            <= GATES["pauroc_difference"]
        )
        report["splits"][split] = result
        print(json.dumps({split: result}), flush=True)
    report["passed"] = all(s["passed"] for s in report["splits"].values())
    write_json(args.output, report)
    print("release_validation", args.model, report["passed"], flush=True)


if __name__ == "__main__":
    main()
