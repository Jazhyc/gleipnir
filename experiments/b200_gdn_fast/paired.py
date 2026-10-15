"""In-server paired A/B of the lean GDN prefill host path (exactness + latency)."""

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
from experiments.b200_monitor_score.run import trial

WORKLOAD = ROOT / "results/b200_context_million/million01/workload.json"
SETTINGS = {"port": 8010, "timeout_seconds": 600}
ARMS = {"fast": True, "upstream": False}
C1 = {
    "synthetic_1k": (1024, 30),
    "synthetic_4k": (4096, 20),
    "synthetic_8k": (8192, 12),
    "synthetic_32k": (32768, 5),
}


def fast_command(command: list[str]) -> list[str]:
    command = command.copy()
    worker = command.index("--worker-cls") + 1
    if not command[worker].endswith("GatedPrecisionWorker"):
        raise ValueError("selected worker changed")
    command[worker] = "experiments.b200_gdn_fast.worker.FastGdnWorker"
    return command


async def run(name: str, rounds: int) -> None:
    out = ROOT / "results/b200_gdn_fast" / name
    out.mkdir(parents=True, exist_ok=False)
    startup_name = f"{name}_server"
    await startup(startup_name, transform=fast_command)
    try:
        workload = {r["prompt_tokens"]: r for r in json.loads(WORKLOAD.read_text())}
        c1 = {case: [workload[n]] * count for case, (n, count) in C1.items()}
        c1["quick64"] = json.loads((DATA / "quick.json").read_text())
        full = json.loads((DATA / "full.json").read_text())
        results = {case: {arm: [] for arm in ARMS} for case in [*c1, "full_c128"]}
        for index in range(rounds):
            order = list(ARMS) if index % 2 == 0 else list(reversed(ARMS))
            for arm in order:
                await rpc("set_gdn_fast", {"enabled": ARMS[arm]})
                for case, rows in c1.items():
                    await sequential(rows[:3])
                    values = await sequential(rows)
                    results[case][arm].append(
                        {
                            "p50_ms": 1000
                            * statistics.median(v["latency_seconds"] for v in values),
                            "scores": [v["score"] for v in values],
                        }
                    )
                await trial(full[:16], 128, SETTINGS)
                values, seconds = await trial(full, 128, SETTINGS)
                results["full_c128"][arm].append(
                    {
                        "input_tokens_per_second": sum(
                            r["prompt_tokens"] for r in values
                        )
                        / seconds,
                        "p50_ms": 1000
                        * statistics.median(r["latency_seconds"] for r in values),
                    }
                )
            write(out / "rounds.json", results)
        await rpc("set_gdn_fast", {"enabled": True})
        summary = {}
        for case, arms in results.items():
            entry = {
                arm: statistics.median(r["p50_ms"] for r in passes)
                for arm, passes in arms.items()
            }
            entry["paired_p50_difference_ms"] = [
                f["p50_ms"] - u["p50_ms"]
                for f, u in zip(arms["fast"], arms["upstream"], strict=True)
            ]
            if case == "full_c128":
                entry["paired_throughput_ratio"] = [
                    f["input_tokens_per_second"] / u["input_tokens_per_second"]
                    for f, u in zip(arms["fast"], arms["upstream"], strict=True)
                ]
            else:
                entry["scores_bitwise_equal"] = all(
                    p["scores"] == arms["upstream"][0]["scores"]
                    for arm in arms
                    for p in arms[arm]
                )
            summary[case] = entry
        summary["audit"] = await rpc("gdn_fast_audit")
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
