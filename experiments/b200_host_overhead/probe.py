"""Measure short C1 host overhead: step timing, then py-spy samples of both servers."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

from experiments.b200_attention_precision.startup import startup
from experiments.b200_inference_benchmark.run import DATA, ROOT, write
from experiments.b200_long_context.run import gpu, rpc
from experiments.b200_monitor_score.run import SERVING, trial

PY_SPY = "/tmp/pyspy/bin/py-spy"
SETTINGS = {"port": 8010, "timeout_seconds": 600}
WORKLOAD = ROOT / "results/b200_context_million/million01/workload.json"


def diagnostic_command(command: list[str]) -> list[str]:
    """Swap only the server module and worker class for attachable variants."""
    command = command.copy()
    if command[1:3] != ["-m", "experiments.b200_attention_precision.server"]:
        raise ValueError("selected server module changed")
    command[2] = "experiments.b200_host_overhead.server"
    worker = command.index("--worker-cls") + 1
    timing = {
        "LastQueryPrecisionWorker": "HostTimingWorker",
        "FastGdnWorker": "HostTimingFastWorker",
    }
    selected = command[worker].rsplit(".", 1)[-1]
    if selected not in timing:
        raise ValueError("selected worker changed")
    command[worker] = f"experiments.b200_host_overhead.worker.{timing[selected]}"
    return command


async def sequential(rows: list[dict], seconds: float | None = None) -> list[dict]:
    """Back-to-back C1 requests over a fixed list, or repeated for a duration."""
    values = []
    started = time.perf_counter()
    index = 0
    while True:
        result, _ = await trial([rows[index % len(rows)]], 1, SETTINGS)
        values.extend(result)
        index += 1
        if seconds is None and index == len(rows):
            return values
        if seconds is not None and time.perf_counter() - started > seconds:
            return values


def latency_summary(values: list[dict]) -> dict:
    latencies = sorted(v["latency_seconds"] * 1000 for v in values)
    return {
        "requests": len(latencies),
        "p50_ms": statistics.median(latencies),
        "p90_ms": latencies[int(0.9 * (len(latencies) - 1))],
        "mean_ms": statistics.fmean(latencies),
    }


def step_summary(steps: list[dict]) -> dict:
    seconds = sorted(s["seconds"] * 1000 for s in steps if s["tokens"] > 0)
    return {
        "steps": len(seconds),
        "p50_ms": statistics.median(seconds),
        "mean_ms": statistics.fmean(seconds),
        "tokens": sorted({s["tokens"] for s in steps if s["tokens"] > 0})[:8],
    }


async def run(name: str, sample_seconds: float, gdn_cp: str) -> None:
    out = ROOT / "results/b200_host_overhead" / name
    out.mkdir(parents=True, exist_ok=False)
    startup_name = f"{name}_server"
    await startup(
        startup_name,
        transform=diagnostic_command,
        environment={"GLEIPNIR_FLASHINFER_GDN_CP": gdn_cp},
    )
    try:
        server = json.loads((SERVING / "server.json").read_text())
        workload = {r["prompt_tokens"]: r for r in json.loads(WORKLOAD.read_text())}
        cases = {
            "synthetic_1k": [workload[1024]] * 50,
            "synthetic_8k": [workload[8192]] * 30,
            "quick64": json.loads((DATA / "quick.json").read_text()),
        }
        report = {"cases": {}}
        for case, rows in cases.items():
            await sequential(rows[:5])
            await rpc("host_timing", {"reset": True})
            values = await sequential(rows)
            timing = await rpc("host_timing", {"reset": True})
            write(out / f"{case}_values.json", values)
            write(out / f"{case}_steps.json", timing)
            report["cases"][case] = {
                "client": latency_summary(values),
                "engine_step": step_summary(timing["steps"]),
            }
            write(out / "summary.json", report)
            print("timed", case, report["cases"][case], flush=True)
        engine = timing["pid"]
        recordings = {}
        for label, pid in (("engine", engine), ("api", server["pid"])):
            target = out / f"pyspy_{label}.txt"
            recordings[label] = subprocess.Popen(
                [
                    PY_SPY,
                    "record",
                    "--pid",
                    str(pid),
                    "--rate",
                    "500",
                    "--duration",
                    str(int(sample_seconds)),
                    "--nonblocking",
                    "--format",
                    "raw",
                    "--output",
                    str(target),
                ],
                stdout=subprocess.DEVNULL,
                stderr=(out / f"pyspy_{label}.log").open("w"),
            )
        await asyncio.sleep(1)
        await rpc("host_timing", {"reset": True})
        sampled = await sequential([workload[1024]], sample_seconds - 2)
        timing = await rpc("host_timing", {"reset": True})
        for process in recordings.values():
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                # py-spy writes its buffered samples on SIGINT, not SIGTERM.
                process.send_signal(signal.SIGINT)
                process.wait(timeout=60)
        report["sampled_1k"] = {
            "client": latency_summary(sampled),
            "engine_step": step_summary(timing["steps"]),
            "pyspy_exit": {k: p.returncode for k, p in recordings.items()},
        }
        write(out / "summary.json", report)
        print("sampled", report["sampled_1k"], flush=True)
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
    parser.add_argument("--sample-seconds", type=float, default=20)
    parser.add_argument("--gdn-cp", choices=["auto", "off"], default="auto")
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("run name must be a stem")
    asyncio.run(run(args.name, args.sample_seconds, args.gdn_cp))


if __name__ == "__main__":
    main()
