"""Run baseline preparation and measurement with durable phase logs/status."""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.local_inference.core import CONFIG, ROOT, write_json


def main() -> None:
    os.environ.update(json.loads(CONFIG.read_text()).get("environment", {}))
    # Kernel builders invoke executables such as ninja by name, even when the
    # runner itself was invoked through an explicit .venv/bin/python path.
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    logs = Path("logs/local/local_inference")
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    for phase in ("prepare", "merge", "reference", "benchmark"):
        status = {"state": "running", "phase": phase, "attempt": stamp}
        write_json(ROOT / "status.json", status)
        path = logs / f"{stamp}-{phase}.log"
        print(f"Starting {phase}; log={path}", flush=True)
        with path.open("x") as handle:
            process = subprocess.run(
                [sys.executable, "-u", "-m", f"experiments.local_inference.{phase}"],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode:
            write_json(
                ROOT / "status.json",
                dict(
                    status,
                    state="failed",
                    returncode=process.returncode,
                    log=str(path),
                ),
            )
            raise SystemExit(process.returncode)
    write_json(ROOT / "status.json", {"state": "complete", "attempt": stamp})


if __name__ == "__main__":
    main()
