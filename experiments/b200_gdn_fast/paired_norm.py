"""In-server A/B of the strided gated norm inside the adopted GDN fast path."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
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
SELECTION = ROOT / "experiments/b200_inference_benchmark/serving_default.json"
SETTINGS = {"port": 8010, "timeout_seconds": 600}
ARMS = {"strided": True, "copy": False}
C1 = {
    "synthetic_8k": (8192, 12),
    "synthetic_16k": (16384, 8),
    "synthetic_32k": (32768, 5),
}
SOURCES = (
    "experiments/b200_last_layer_trim/worker.py",
    "src/gleipnir/serving/last_query_attention.py",
    "experiments/b200_gdn_route/worker.py",
    "src/gleipnir/serving/gdn_route.py",
    "experiments/b200_gdn_fast/worker.py",
    "src/gleipnir/serving/gdn_fast.py",
)


def stack_command(command: list[str]) -> list[str]:
    """Base recipe plus the adopted stack, binding current (edited) sources."""
    selection = json.loads(SELECTION.read_text())
    command = command.copy()
    worker = command.index("--worker-cls") + 1
    if not command[worker].endswith("AttentionPrecisionWorker"):
        raise ValueError("base worker changed")
    command[worker] = "experiments.b200_gdn_fast.worker.FastGdnWorker"
    index = command.index("--additional-config") + 1
    additional = json.loads(command[index])
    for path in SOURCES:
        additional["gleipnir_frost_fp4"][path] = hashlib.sha256(
            (ROOT / path).read_bytes()
        ).hexdigest()
    receipt = selection["last_query_attention"]["validation"]
    additional["last_query_attention"] = {
        "validation": receipt,
        "validation_sha256": selection["artifact_bindings"][receipt],
    }
    additional["gdn_route"] = {"cp_min_tokens": selection["gdn_route"]["cp_min_tokens"]}
    command[index] = json.dumps(additional, sort_keys=True)
    return command


async def run(name: str, rounds: int) -> None:
    out = ROOT / "results/b200_gdn_fast" / name
    out.mkdir(parents=True, exist_ok=False)
    startup_name = f"{name}_server"
    await startup(startup_name, last_query=False, transform=stack_command)
    try:
        audits = {
            "trim": await rpc("last_query_audit"),
            "route": await rpc("gdn_route_audit"),
            "fast": await rpc("gdn_fast_audit"),
        }
        write(out / "startup_audits.json", audits)
        if not audits["fast"]["enabled"] or audits["route"]["cp_min_tokens"] != 8192:
            raise ValueError("adopted stack not active")
        workload = {r["prompt_tokens"]: r for r in json.loads(WORKLOAD.read_text())}
        c1 = {case: [workload[n]] * count for case, (n, count) in C1.items()}
        c1["quick64"] = json.loads((DATA / "quick.json").read_text())
        full = json.loads((DATA / "full.json").read_text())
        results = {case: {arm: [] for arm in ARMS} for case in [*c1, "full_c128"]}
        for index in range(rounds):
            order = list(ARMS) if index % 2 == 0 else list(reversed(ARMS))
            for arm in order:
                await rpc("set_gdn_fast", {"enabled": True, "strided_norm": ARMS[arm]})
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
        await rpc("set_gdn_fast", {"enabled": True, "strided_norm": False})
        summary = {}
        for case, arms in results.items():
            entry = {
                arm: statistics.median(r["p50_ms"] for r in p)
                for arm, p in arms.items()
            }
            entry["paired_p50_difference_ms"] = [
                s["p50_ms"] - c["p50_ms"]
                for s, c in zip(arms["strided"], arms["copy"], strict=True)
            ]
            if case == "full_c128":
                entry["paired_throughput_ratio"] = [
                    s["input_tokens_per_second"] / c["input_tokens_per_second"]
                    for s, c in zip(arms["strided"], arms["copy"], strict=True)
                ]
            else:
                entry["scores_bitwise_equal"] = all(
                    p["scores"] == arms["copy"][0]["scores"]
                    for a in arms
                    for p in arms[a]
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
