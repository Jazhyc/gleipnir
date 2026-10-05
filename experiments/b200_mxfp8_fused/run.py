"""Launch bounded fused preparation diagnostics with existing shared caches."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from experiments.b200_nvidia_mxfp8.run import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--square", action="store_true")
    args = parser.parse_args()
    if not args.attempt.isalnum():
        raise ValueError("attempt must be alphanumeric")
    cfg = yaml.safe_load(Path("experiments/b200_mxfp8_fused/config.yaml").read_text())
    root = Path("results/b200_mxfp8_fused") / args.attempt
    log = Path(cfg["logs"]) / f"{args.attempt}.log"
    if root.exists() or log.exists():
        raise ValueError("preserve existing receipts")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "experiments.b200_mxfp8_fused.kernel_canary",
        "--output",
        str(root),
    ]
    if args.square:
        command.append("--square")
    env = {**os.environ, **environment(cfg)}
    with log.open("x") as handle:
        result = subprocess.run(
            command, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
    (root / "launch.json").write_text(
        json.dumps(
            {"returncode": result.returncode, "command": command, "config": cfg},
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
