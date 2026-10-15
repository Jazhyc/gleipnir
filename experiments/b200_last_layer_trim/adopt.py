"""Same-node A/B of the selected default with and without last-query attention.

Launch each recipe through the normal startup (canary-guarded), then time real
quick64 prompts at c1 and full320 at c128. C1 scores must be bitwise equal; c128
batch composition varies between passes, so its scores are compared as a
distribution. Retire each server before the next. Never changes capacity.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
from pathlib import Path

from experiments.b200_attention_precision.startup import startup
from experiments.b200_inference_benchmark.run import DATA, ROOT, write
from experiments.b200_long_context.run import gpu
from experiments.b200_monitor_score.run import trial

SETTINGS = {"port": 8010, "timeout_seconds": 600}
WORKLOADS = (("quick", 1, 3), ("full", 128, 3))


async def measure(out: Path) -> dict:
    summary = {}
    for workload, concurrency, repeats in WORKLOADS:
        rows = json.loads((DATA / f"{workload}.json").read_text())
        await trial(rows, concurrency, SETTINGS)
        passes = []
        for repeat in range(repeats):
            values, seconds = await trial(rows, concurrency, SETTINGS)
            write(out / f"{workload}_c{concurrency}_repeat{repeat}.json", values)
            tokens = sum(r["prompt_tokens"] for r in values)
            latencies = sorted(r["latency_seconds"] for r in values)
            passes.append(
                {
                    "seconds": seconds,
                    "input_tokens_per_second": tokens / seconds,
                    "requests_per_second": len(values) / seconds,
                    "p50_latency_seconds": statistics.median(latencies),
                    "p95_latency_seconds": latencies[int(0.95 * (len(latencies) - 1))],
                    "scores": [r["score"] for r in values],
                }
            )
        summary[f"{workload}_c{concurrency}"] = passes
    return summary


def retire(name: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "experiments.b200_vllm031.stop", "--archive-name", name],
        cwd=ROOT,
        check=True,
    )
    if gpu()["apps"]:
        raise ValueError("GPU not empty after verified server retirement")


def compare(control: dict, candidate: dict) -> dict:
    result = {}
    for key in control:
        a, b = control[key], candidate[key]
        throughput = [
            statistics.median(p["input_tokens_per_second"] for p in x) for x in (a, b)
        ]
        p50 = [statistics.median(p["p50_latency_seconds"] for p in x) for x in (a, b)]
        result[key] = {
            "control_input_tokens_per_second": throughput[0],
            "candidate_input_tokens_per_second": throughput[1],
            "throughput_ratio": throughput[1] / throughput[0],
            "control_p50_latency_seconds": p50[0],
            "candidate_p50_latency_seconds": p50[1],
            "control_ranges": [
                min(p["input_tokens_per_second"] for p in a),
                max(p["input_tokens_per_second"] for p in a),
            ],
            "candidate_ranges": [
                min(p["input_tokens_per_second"] for p in b),
                max(p["input_tokens_per_second"] for p in b),
            ],
            "scores_bitwise_equal_all_passes": all(
                p["scores"] == q["scores"] for p in a for q in b
            ),
            "max_score_difference": max(
                abs(x - y)
                for p in a
                for q in b
                for x, y in zip(p["scores"], q["scores"], strict=True)
            ),
        }
    return result


async def run(name: str) -> None:
    out = ROOT / "results/b200_last_layer_trim" / name
    out.mkdir(parents=True, exist_ok=False)
    results = {}
    for label, last_query in (("control", False), ("adopted", True)):
        startup_name = f"{name}_{label}"
        await startup(startup_name, last_query=last_query)
        try:
            results[label] = await measure(out / label)
        finally:
            retire(startup_name)
        write(out / f"{label}.json", results[label])
    report = {"comparison": compare(results["control"], results["adopted"])}
    write(out / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("run name must be a stem")
    asyncio.run(run(args.name))


if __name__ == "__main__":
    main()
