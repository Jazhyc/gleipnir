#!/usr/bin/env python3
"""Wait for the 9B scaling scheduler, then start the queued mixed 4B run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.tool_trajectory_monitoring.run_distillation_lambda import (
    atomic_write_json,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCALING_STATUS = Path(
    "results/tool_trajectory_distillation_scaling/lambda_status.json"
)
DEFAULT_FOLLOWUP_STATUS = Path(
    "results/tool_trajectory_distillation_mixed_qwen4b/lambda_status.json"
)


def process_matches(pid: int, marker: str) -> bool:
    """Return whether ``pid`` still names the prerequisite scheduler process."""
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    return marker.encode() in command


def read_status(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prerequisite-pid", type=int, required=True)
    parser.add_argument(
        "--prerequisite-marker", default="run_distillation_lambda"
    )
    parser.add_argument(
        "--scaling-status", type=Path, default=DEFAULT_SCALING_STATUS
    )
    parser.add_argument(
        "--followup-status", type=Path, default=DEFAULT_FOLLOWUP_STATUS
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--revision", default=os.environ.get("GLEIPNIR_COMMIT"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    queued_at = time.time()
    atomic_write_json(
        args.followup_status.resolve(),
        {
            "state": "queued",
            "phase": "waiting_for_9b_campaign",
            "queued_at_unix": queued_at,
            "prerequisite_pid": args.prerequisite_pid,
            "prerequisite_marker": args.prerequisite_marker,
            "revision": args.revision,
        },
    )
    while process_matches(args.prerequisite_pid, args.prerequisite_marker):
        time.sleep(args.poll_seconds)
    prerequisite = read_status(args.scaling_status.resolve())
    if prerequisite.get("state") != "complete":
        atomic_write_json(
            args.followup_status.resolve(),
            {
                "state": "blocked_on_prerequisite",
                "phase": "not_started",
                "queued_at_unix": queued_at,
                "checked_at_unix": time.time(),
                "prerequisite_status": prerequisite,
                "revision": args.revision,
            },
        )
        raise RuntimeError("9B scaling campaign did not finish successfully")
    command = [
        sys.executable,
        "-m",
        "experiments.tool_trajectory_monitoring.run_mixed_4b_lambda",
    ]
    if args.revision:
        command.extend(["--revision", args.revision])
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
