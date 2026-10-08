"""Sweep the current FP8 scorer on a new host without changing the recipe."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from gleipnir.serving.benchmark import (
    measurement_summary,
    paired_score_summary,
    ranking_comparison,
)
from gleipnir.serving.reference import selected_serving_default

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).parent
CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128)


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def validate_contract(settings: dict, rows: list[dict], references: list[list]) -> None:
    """Reject changed cohorts, incomplete controls and recipe selection."""
    if (
        tuple(settings["concurrencies"]) != CONCURRENCIES
        or settings["timed_repeats"] != 3
        or settings["warmup_repeats"] != 1
        or settings["promote"]
        or len(rows) != 320
        or settings["rows"] != 320
        or sum(r["prompt_tokens"] for r in rows) != 1_310_581
        or settings["prompt_tokens"] != 1_310_581
        or len(references) != 6
    ):
        raise ValueError("current scaling workload/repeat contract changed")
    identity = [(r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in rows]
    if len({r[0] for r in identity}) != len(identity) or any(
        [(r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in run] != identity
        for run in references
    ):
        raise ValueError("current scaling reference identity changed")
    paired_score_summary(references, references)


async def run(name: str, expected_gpu_uuid: str) -> None:
    from experiments.b200_attention_precision.startup import startup
    from experiments.b200_long_context.run import gpu
    from experiments.b200_monitor_score.run import SERVING, trial
    from experiments.b200_vllm031.run import archive_audits

    settings = json.loads((EXPERIMENT / "current_config.json").read_text())
    workload = ROOT / settings["workload"]
    if hashlib.sha256(workload.read_bytes()).hexdigest() != settings["workload_sha256"]:
        raise ValueError("current scaling workload checksum changed")
    references = []
    for path, digest in settings["reference_sha256"].items():
        value = ROOT / path
        if hashlib.sha256(value.read_bytes()).hexdigest() != digest:
            raise ValueError(f"FP8 comparison reference checksum changed: {path}")
        references.append(json.loads(value.read_text()))
    rows = json.loads(workload.read_text())
    validate_contract(settings, rows, references)
    selection, command = selected_serving_default(ROOT)
    for flag, value in (
        ("--max-num-seqs", "128"),
        ("--max-num-batched-tokens", "32768"),
        ("--max-model-len", "32768"),
    ):
        if command[command.index(flag) + 1] != value:
            raise ValueError("current scaling engine limits changed")
    if "--no-enable-prefix-caching" not in command:
        raise ValueError("current scaling prefix cache policy changed")
    snapshot = gpu()
    if snapshot["apps"] or expected_gpu_uuid not in snapshot["gpu"]:
        raise ValueError("current scaling requires the verified idle target GPU")
    if (SERVING / "server.json").exists():
        raise ValueError("current scaling requires no previous scorer")
    out = ROOT / "results/b200_score_scaling" / name
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "trials": [], "promoted": False}
    write(out / "settings.json", settings | {"expected_gpu_uuid": expected_gpu_uuid})
    write(out / "selection.json", selection)
    write(out / "initial_gpu.json", snapshot)
    write(
        out / "host.json",
        {
            "cpu": next(
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
            "memory_limit": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
        },
    )
    for path in (
        Path(__file__),
        EXPERIMENT / "current_config.json",
        EXPERIMENT / "README.md",
    ):
        target = out / "executed_sources" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    write(out / "summary.json", report)
    try:
        await startup(name + "_startup")
        server = json.loads((SERVING / "server.json").read_text())
        write(out / "server.json", server)
        write(
            out / "startup_canary.json",
            json.loads(
                (
                    ROOT
                    / "results/b200_attention_precision"
                    / (name + "_startup")
                    / "canary.json"
                ).read_text()
            ),
        )
        report["status"] = "measuring"
        write(out / "summary.json", report)
        for concurrency in CONCURRENCIES:
            warm, _ = await trial(rows, concurrency, settings)
            write(out / f"c{concurrency}_warmup.json", warm)
            observations = []
            for repeat in range(settings["timed_repeats"]):
                values, seconds = await trial(rows, concurrency, settings)
                observations.append(values)
                write(out / f"c{concurrency}_repeat{repeat}.json", values)
                measured = {
                    "concurrency": concurrency,
                    "repeat": repeat,
                    **measurement_summary(values, seconds),
                }
                report["trials"].append(measured)
                write(out / "summary.json", report)
                print(
                    "current_scaling_pass",
                    concurrency,
                    repeat,
                    measured["prompt_tokens_per_second"],
                    measured["latency"],
                    flush=True,
                )
            write(
                out / f"c{concurrency}_comparison.json",
                {
                    "scores": paired_score_summary(references, observations),
                    "ranking": ranking_comparison(rows, references, observations),
                    "reference": settings["reference_results"],
                },
            )
        archive_audits(out)
        report.update(status="complete", server_retained_warm=True)
        write(out / "summary.json", report)
        write(out / "closure.json", gpu())
        print("current_scaling_complete", name, flush=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write(out / "summary.json", report)
        if (SERVING / "server.json").exists():
            active = json.loads((SERVING / "server.json").read_text())
            if active.get("log", "").endswith(f"/{name}_startup_server.log"):
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "experiments.b200_vllm031.stop",
                        "--archive-name",
                        name + "_failed",
                    ],
                    check=True,
                )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()
    if args.name in {"", ".", ".."} or Path(args.name).name != args.name:
        raise ValueError("run name must be a stem")
    asyncio.run(run(args.name, args.expected_gpu_uuid))


if __name__ == "__main__":
    main()
