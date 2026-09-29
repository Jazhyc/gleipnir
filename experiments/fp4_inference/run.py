"""Run one frozen serving condition inside the time-bounded GPU allocation."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from experiments.local_inference.core import write_json

STOP_UTC = datetime.datetime(2026, 9, 29, 23, 29, 21, tzinfo=datetime.UTC)
CONFIGS = Path(__file__).with_name("configs")
ROOT = Path("results/fp4_inference")


def record_telemetry(output: Path, stopped: threading.Event) -> None:
    """Log GPU measurements while the worker runs; this is not an agent heartbeat."""
    fields = (
        "temperature.gpu,clocks.current.sm,memory.used,utilization.gpu,power.draw,"
        "clocks_throttle_reasons.sw_thermal_slowdown,"
        "clocks_throttle_reasons.hw_thermal_slowdown"
    )
    while not stopped.is_set():
        status = output / "status.json"
        if status.is_file():
            try:
                values = (
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            f"--query-gpu={fields}",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=5,
                    )
                    .strip()
                    .split(", ")
                )
                sample = {
                    "utc": datetime.datetime.now(datetime.UTC).isoformat(),
                    "status": json.loads(status.read_text()),
                    "gpu": dict(zip(fields.split(","), values, strict=True)),
                }
                with (output / "gpu_telemetry.jsonl").open("a") as handle:
                    handle.write(json.dumps(sample) + "\n")
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                print(f"Telemetry observation failed: {error}", flush=True)
            stopped.wait(5)
        else:
            stopped.wait(1)


def remaining_seconds(now: datetime.datetime | None = None) -> float:
    """Return usable time before the user-requested ten-minute reserve."""
    return (STOP_UTC - (now or datetime.datetime.now(datetime.UTC))).total_seconds()


def condition_config(name: str) -> Path:
    """Resolve a named checked-in condition without path traversal."""
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
        raise ValueError(
            "Condition names must use lowercase letters, digits, underscores"
        )
    path = CONFIGS / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    args = parser.parse_args()
    path = condition_config(args.condition)
    config = json.loads(path.read_text())
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError(f"Preserve existing output: {output}")
    budget = remaining_seconds()
    if budget < 120:
        raise RuntimeError("Insufficient usable time before the campaign GPU deadline")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(config.get("environment", {}))
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    started = time.perf_counter()
    sources = [Path(__file__), Path("experiments/local_inference/benchmark.py")]
    sources.extend(Path(p) for p in config.get("additional_code_files", []))
    code_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    command = [
        sys.executable,
        "-u",
        "-m",
        "experiments.local_inference.run",
        "--config",
        str(path),
    ]
    process = subprocess.Popen(command, start_new_session=True)
    stopped = threading.Event()
    observer = threading.Thread(
        target=record_telemetry, args=(output, stopped), daemon=True
    )
    observer.start()
    timed_out = False
    try:
        returncode = process.wait(timeout=budget)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        returncode = 124
    finally:
        stopped.set()
        observer.join(timeout=6)
    write_json(
        ROOT / f"{args.condition}_execution.json",
        {
            "condition": args.condition,
            "returncode": returncode,
            "deadline_utc": STOP_UTC.isoformat(),
            "deadline_reached": timed_out,
            "seconds": time.perf_counter() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "code_sha256": code_hashes,
        },
    )
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
