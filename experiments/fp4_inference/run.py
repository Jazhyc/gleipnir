"""Run one frozen serving condition inside the time-bounded GPU allocation."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from experiments.local_inference.core import write_json

STOP_UTC = datetime.datetime(2026, 9, 29, 23, 29, 21, tzinfo=datetime.UTC)
CONFIGS = Path(__file__).with_name("configs")
ROOT = Path("results/fp4_inference")


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
    command = [
        sys.executable, "-u", "-m", "experiments.local_inference.run",
        "--config", str(path),
    ]
    process = subprocess.Popen(command, start_new_session=True)
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
    write_json(
        ROOT / f"{args.condition}_execution.json",
        {
            "condition": args.condition,
            "returncode": returncode,
            "deadline_utc": STOP_UTC.isoformat(),
            "deadline_reached": timed_out,
            "seconds": time.perf_counter() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
