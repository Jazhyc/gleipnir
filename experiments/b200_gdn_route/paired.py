"""Paired in-server A/B of GDN routes for short C1 requests (diagnostic)."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
from pathlib import Path

from experiments.b200_attention_precision.startup import startup
from experiments.b200_host_overhead.probe import sequential
from experiments.b200_inference_benchmark.run import DATA, ROOT, write
from experiments.b200_long_context.run import gpu, rpc

WORKLOAD = ROOT / "results/b200_context_million/million01/workload.json"
# Threshold 1: every call keeps FlashInfer's auto (CP for one request).
# Threshold 32769: every chunk (<=32K) takes non-CP.
ARMS = {"auto": 1, "non_cp": 32769}


def gated_command(command: list[str]) -> list[str]:
    command = command.copy()
    worker = command.index("--worker-cls") + 1
    if not command[worker].endswith("LastQueryPrecisionWorker"):
        raise ValueError("selected worker changed")
    command[worker] = "experiments.b200_gdn_route.worker.GatedPrecisionWorker"
    return command


CASES = {
    "short": {"synthetic_1k": (1024, 30), "synthetic_4k": (4096, 20)},
    "crossover": {
        "synthetic_8k": (8192, 12),
        "synthetic_16k": (16384, 8),
        "synthetic_32k": (32768, 5),
    },
}


async def run(name: str, rounds: int, cases_name: str) -> None:
    out = ROOT / "results/b200_gdn_route" / name
    out.mkdir(parents=True, exist_ok=False)
    startup_name = f"{name}_server"
    await startup(
        startup_name,
        transform=gated_command,
        environment={"GLEIPNIR_GDN_CP_MIN_TOKENS": "8192"},
    )
    try:
        workload = {r["prompt_tokens"]: r for r in json.loads(WORKLOAD.read_text())}
        cases = {
            case: [workload[length]] * count
            for case, (length, count) in CASES[cases_name].items()
        }
        if cases_name == "short":
            cases["quick64"] = json.loads((DATA / "quick.json").read_text())
        results = {case: {arm: [] for arm in ARMS} for case in cases}
        for round_index in range(rounds):
            order = list(ARMS) if round_index % 2 == 0 else list(reversed(ARMS))
            for arm in order:
                await rpc("set_gdn_route", {"min_tokens": ARMS[arm]})
                for case, rows in cases.items():
                    await sequential(rows[:3])
                    values = await sequential(rows)
                    results[case][arm].append(
                        {
                            "p50_ms": 1000
                            * statistics.median(v["latency_seconds"] for v in values),
                            "scores": [v["score"] for v in values],
                        }
                    )
            write(out / "rounds.json", results)
        summary = {}
        for case, arms in results.items():
            medians = {
                a: statistics.median(r["p50_ms"] for r in v) for a, v in arms.items()
            }
            pairs = [
                a["p50_ms"] - b["p50_ms"]
                for a, b in zip(arms["non_cp"], arms["auto"], strict=True)
            ]
            summary[case] = {
                "median_p50_ms": medians,
                "paired_difference_ms": pairs,
                "non_cp_faster_rounds": sum(d < 0 for d in pairs),
                "scores_stable_within_arm": all(
                    r["scores"] == arms[a][0]["scores"] for a in arms for r in arms[a]
                ),
            }
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
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--cases", choices=sorted(CASES), default="short")
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("run name must be a stem")
    asyncio.run(run(args.name, args.rounds, args.cases))


if __name__ == "__main__":
    main()
