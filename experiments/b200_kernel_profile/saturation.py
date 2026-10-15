"""Where saturated batches leave the GPU idle: between steps or inside them.

Launch the adopted default through normal startup with the step-timing worker,
then for 2K windows at c32 and full320 at c128: one excluded warmup, one
untraced pass with per-step host timing, and one torch-profiled pass. The
trace's CPU step annotations (`execute_context_*`) and kernels share a clock,
so each GPU idle gap is assigned to input preparation (step start to first
kernel), step tail (last kernel to step return) or the engine loop between
steps (scheduler, IPC, output processing). Diagnostic only; profiling perturbs
host time, which is why the untraced pass is kept separately.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import statistics
import subprocess
import sys
from pathlib import Path

from experiments.b200_attention_precision.startup import startup
from experiments.b200_host_overhead.probe import diagnostic_command
from experiments.b200_inference_benchmark.run import ROOT, write
from experiments.b200_kernel_profile.analyze import merged_busy
from experiments.b200_kernel_profile.profile import profiled
from experiments.b200_long_context.run import gpu, rpc
from experiments.b200_monitor_score.run import trial

SETTINGS = {"port": 8010, "timeout_seconds": 600}
CASES = (
    ("twok_c32", "results/b200_context_scaling/twok01/workload.json", 32),
    ("full320_c128", "data/b200_inference_benchmark/full.json", 128),
)


def untraced(steps: list[dict], seconds: float) -> dict:
    """Engine-side split of one pass: inside execute_model versus between."""
    steps = [s for s in steps if s["tokens"] > 0]
    inside = sum(s["seconds"] for s in steps)
    between = [
        b["started"] - (a["started"] + a["seconds"])
        for a, b in zip(steps, steps[1:], strict=False)
    ]
    span = steps[-1]["started"] + steps[-1]["seconds"] - steps[0]["started"]
    return {
        "client_seconds": seconds,
        "steps": len(steps),
        "median_step_tokens": statistics.median(s["tokens"] for s in steps),
        "median_execute_ms": 1000 * statistics.median(s["seconds"] for s in steps),
        "engine_span_seconds": span,
        "execute_share_of_span": inside / span,
        "between_steps_share_of_span": sum(between) / span,
        "median_between_ms": 1000 * statistics.median(between) if between else None,
    }


def gap_split(trace: Path) -> dict:
    """Split GPU idle time into within-step and between-step gaps.

    GPU-side `execute_context_*` annotations give each step's GPU extent; the
    CPU-side annotation of the same step covers only its launches. A gap
    between steps is split at the next step's CPU start: before it, the host
    is finishing the previous step (sync, output processing, scheduling);
    after it, the host is preparing inputs and launching the first kernels.
    """
    with gzip.open(trace, "rt") as handle:
        events = json.load(handle)["traceEvents"]

    def extents(category: str) -> list[list[float]]:
        marks = [
            e
            for e in events
            if e.get("cat") == category
            and e.get("name", "").startswith("execute_context_")
        ]
        # GPU annotations repeat as zero-length markers on a second stream;
        # keep the thread whose annotations span the steps.
        totals: dict = {}
        for e in marks:
            totals[e.get("tid")] = totals.get(e.get("tid"), 0) + e["dur"]
        main = max(totals, key=totals.get)
        merged: list[list[float]] = []
        for start, end in sorted(
            (e["ts"], e["ts"] + e["dur"]) for e in marks if e.get("tid") == main
        ):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return merged

    gpu_steps, cpu_steps = extents("gpu_user_annotation"), extents("user_annotation")
    kernels = [
        (e["ts"], e["ts"] + e["dur"])
        for e in events
        if e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
        and e.get("dur", 0) > 0
    ]
    busy, window = merged_busy(list(kernels))
    within, early = 0.0, 0.0
    for (start, end), (_, cpu_end) in zip(gpu_steps, cpu_steps, strict=False):
        inside = sorted(k for k in kernels if start <= k[0] < end)
        if not inside:
            continue
        step_busy, _ = merged_busy(list(inside))
        within += (end - start) - step_busy
        # Idle while the CPU is still launching this step: the GPU has caught
        # up with the launch queue (launch-bound phase).
        reach = inside[0][1]
        for k_start, k_end in inside[1:]:
            if k_start > reach and k_start <= cpu_end:
                early += k_start - reach
            reach = max(reach, k_end)
    host_finish, host_prepare = 0.0, 0.0
    for (_, end), (start, _) in zip(gpu_steps, gpu_steps[1:], strict=False):
        cpu_start = next((c[0] for c in cpu_steps if end <= c[0] <= start), start)
        host_finish += cpu_start - end
        host_prepare += start - cpu_start
    covered = gpu_steps[-1][1] - gpu_steps[0][0]
    parts = {
        "within_steps": within,
        "within_steps_while_cpu_launching": early,
        "between_steps_finish_and_schedule": host_finish,
        "between_steps_prepare_and_launch": host_prepare,
    }
    return {
        "steps": len(gpu_steps),
        "gpu_idle_share_of_window": 1 - busy / window,
        "steps_window_ms": covered / 1000,
        "median_gpu_step_ms": statistics.median(e - s for s, e in gpu_steps) / 1000,
        "idle_ms": {k: v / 1000 for k, v in parts.items()},
        "idle_share_of_steps_window": {k: v / covered for k, v in parts.items()},
    }


async def run(name: str) -> None:
    out = ROOT / "results/b200_kernel_profile" / name
    out.mkdir(parents=True, exist_ok=False)
    startup_name = f"{name}_server"
    await startup(startup_name, transform=diagnostic_command)
    report: dict = {"status": "running", "cases": {}}
    try:
        for case, workload, concurrency in CASES:
            rows = json.loads((ROOT / workload).read_text())
            await trial(rows, concurrency, SETTINGS)
            await rpc("host_timing", {"reset": True})
            values, seconds = await trial(rows, concurrency, SETTINGS)
            steps = (await rpc("host_timing", {"reset": True}))["steps"]
            write(out / f"{case}_steps.json", steps)
            entry = {
                "concurrency": concurrency,
                "untraced": untraced(steps, seconds)
                | {
                    "tokens_per_second": sum(v["prompt_tokens"] for v in values)
                    / seconds
                },
            }
            traced = await profiled(rows, concurrency, SETTINGS, out / case)
            trace = next((out / case).glob("*.gz"))
            entry["traced"] = {
                k: traced[k] for k in traced if k not in {"top_kernels"}
            } | {"gaps": gap_split(trace)}
            report["cases"][case] = entry
            write(out / "summary.json", report)
            print(
                "saturation",
                case,
                json.dumps(entry["untraced"]),
                json.dumps(entry["traced"]["gaps"]),
                flush=True,
            )
        report["status"] = "complete"
        write(out / "summary.json", report)
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
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("run name must be a stem")
    asyncio.run(run(args.name))


if __name__ == "__main__":
    main()
