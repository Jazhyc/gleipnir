"""Same-server A/B of the adopted default with the GDN gate on versus auto."""

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
from experiments.b200_long_context.run import gpu, rpc
from experiments.b200_monitor_score.run import trial

SETTINGS = {"port": 8010, "timeout_seconds": 600}
ARMS = {"gated": 8192, "auto": 1}
WORKLOADS = (("quick", 1), ("full", 128))


async def run(name: str, rounds: int) -> None:
    out = ROOT / "results/b200_gdn_route" / name
    out.mkdir(parents=True, exist_ok=False)
    startup_name = f"{name}_server"
    await startup(startup_name)
    try:
        audit = await rpc("gdn_route_audit")
        if audit["cp_min_tokens"] != ARMS["gated"]:
            raise ValueError("adopted default does not carry the selected gate")
        results = {w: {a: [] for a in ARMS} for w, _ in WORKLOADS}
        for workload, concurrency in WORKLOADS:
            rows = json.loads((DATA / f"{workload}.json").read_text())
            for index in range(rounds):
                order = list(ARMS) if index % 2 == 0 else list(reversed(ARMS))
                for arm in order:
                    await rpc("set_gdn_route", {"min_tokens": ARMS[arm]})
                    await trial(rows[:8], concurrency, SETTINGS)
                    values, seconds = await trial(rows, concurrency, SETTINGS)
                    results[workload][arm].append(
                        {
                            "seconds": seconds,
                            "input_tokens_per_second": sum(
                                r["prompt_tokens"] for r in values
                            )
                            / seconds,
                            "p50_ms": 1000
                            * statistics.median(r["latency_seconds"] for r in values),
                            "scores": [r["score"] for r in values],
                        }
                    )
                write(out / "rounds.json", results)
        await rpc("set_gdn_route", {"min_tokens": ARMS["gated"]})
        summary = {}
        for workload, arms in results.items():
            summary[workload] = {
                arm: {
                    "median_input_tokens_per_second": statistics.median(
                        r["input_tokens_per_second"] for r in passes
                    ),
                    "median_p50_ms": statistics.median(r["p50_ms"] for r in passes),
                }
                for arm, passes in arms.items()
            }
            summary[workload]["paired_p50_difference_ms"] = [
                g["p50_ms"] - a["p50_ms"]
                for g, a in zip(arms["gated"], arms["auto"], strict=True)
            ]
            summary[workload]["paired_throughput_ratio"] = [
                g["input_tokens_per_second"] / a["input_tokens_per_second"]
                for g, a in zip(arms["gated"], arms["auto"], strict=True)
            ]
        summary["audit"] = await rpc("gdn_route_audit")
        write(out / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.b200_vllm031.stop",
                "--archive-name",
                startup_name,
            ],
            cwd=ROOT,
            check=True,
        )
        if gpu()["apps"]:
            raise ValueError("GPU not empty after verified server retirement")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("run name must be a stem")
    asyncio.run(run(args.name, args.rounds))


if __name__ == "__main__":
    main()
