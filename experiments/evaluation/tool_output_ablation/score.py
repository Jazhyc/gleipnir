"""Score a stripped OOD workload against a running release server.

Batches like the recorded campaign (``experiments/release_serving/validate.py``):
workload order, sequential batches of 128 rows at concurrency 128, a fresh
client per batch. Writes the scored rows and a timing summary.

``python -m experiments.evaluation.tool_output_ablation.score --model 4b --name run01``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics

from experiments.evaluation.tool_output_ablation.count import CONFIG, ROOT
from gleipnir.data.monitoring import read_rows, write_json, write_rows
from gleipnir.evaluation.http_score import score_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(CONFIG["models"]), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    workload = read_rows(ROOT / CONFIG["stripped_dir"] / f"{args.model}_ood.jsonl")
    config = {
        "port": args.port,
        "concurrency": CONFIG["concurrency"],
        "timeout_seconds": 600,
    }
    scored, seconds = [], 0.0
    step = CONFIG["batch_rows"]
    for start in range(0, len(workload), step):
        values, elapsed = asyncio.run(
            score_batch(workload[start : start + step], config)
        )
        scored += values
        seconds += elapsed
    out = ROOT / CONFIG["results_dir"] / args.name
    write_rows(
        out / f"{args.model}_ood.jsonl",
        [
            {
                "id": r["id"],
                "prompt_sha256": r["prompt_sha256"],
                "prompt_tokens": r["prompt_tokens"],
                "score": r["score"],
                "margin": r["margin"],
                "latency_seconds": r["latency_seconds"],
            }
            for r in scored
        ],
    )
    tokens = sum(r["prompt_tokens"] for r in workload)
    summary = {
        "model": args.model,
        "rows": len(scored),
        "seconds": seconds,
        "prompt_tokens": tokens,
        "prompt_tokens_per_second": tokens / seconds,
        "median_latency_seconds": statistics.median(
            r["latency_seconds"] for r in scored
        ),
    }
    write_json(out / f"{args.model}_timing.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
