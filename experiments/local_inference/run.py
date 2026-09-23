"""Run baseline preparation and measurement with durable phase logs/status."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.local_inference.core import CONFIG, ROOT, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration32", action="store_true")
    args = parser.parse_args()
    config_path = CONFIG.with_name("iteration32.json") if args.iteration32 else CONFIG
    config = json.loads(config_path.read_text())
    os.environ.update(config.get("environment", {}))
    # Kernel builders invoke executables such as ninja by name, even when the
    # runner itself was invoked through an explicit .venv/bin/python path.
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    logs = Path("logs/local/local_inference")
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    phases = (
        ("prepare32", "benchmark")
        if args.iteration32
        else ("prepare", "merge", "reference", "benchmark")
    )
    status_path = ROOT / (
        "iteration32_status.json" if args.iteration32 else "status.json"
    )
    for phase in phases:
        if phase == "benchmark":
            output = Path(config.get("output", ROOT / "baseline"))
            if output.exists():
                raise FileExistsError(f"Preserve existing benchmark output: {output}")
        status = {"state": "running", "phase": phase, "attempt": stamp}
        write_json(status_path, status)
        path = logs / f"{stamp}-{phase}.log"
        print(f"Starting {phase}; log={path}", flush=True)
        command = [sys.executable, "-u", "-m", f"experiments.local_inference.{phase}"]
        if phase == "benchmark":
            command.extend(["--config", str(config_path)])
        started = time.perf_counter()
        with path.open("x") as handle:
            process = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if phase == "benchmark":
            output = Path(config.get("output", ROOT / "baseline"))
            write_json(
                output / "process_timing.json",
                {
                    "seconds": time.perf_counter() - started,
                    "returncode": process.returncode,
                    "includes": (
                        "Python startup, validation, engine initialization, warmup, "
                        "scoring, persistence, and shutdown"
                    ),
                },
            )
        if process.returncode:
            write_json(
                status_path,
                dict(
                    status,
                    state="failed",
                    returncode=process.returncode,
                    log=str(path),
                ),
            )
            raise SystemExit(process.returncode)
    write_json(status_path, {"state": "complete", "attempt": stamp})


if __name__ == "__main__":
    main()
