"""Launch a receipt-preserving native packed MXFP8 check with shared caches."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from experiments.b200_nvidia_mxfp8.run import environment

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/b200_nvidia_mxfp8_varlen/config.yaml",
    )
    args = parser.parse_args()
    if not args.attempt.isalnum():
        raise ValueError("attempt must be an alphanumeric receipt name")
    config = yaml.safe_load(args.config.read_text())
    output = ROOT / "results/b200_nvidia_mxfp8_varlen" / args.attempt
    log = ROOT / config["logs"] / (args.attempt + ".log")
    if output.exists() or log.exists():
        raise ValueError("preserve existing attempt outputs and logs")
    output.mkdir(parents=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "starting", "config": config}
    receipt = output / "launch.json"
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    env = {**os.environ, **environment(config)}
    with log.open("x") as handle:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.b200_nvidia_mxfp8_varlen.kernel_canary",
                "--output",
                str(output / "kernel_canary.json"),
            ],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    report.update(
        status="complete" if result.returncode == 0 else "failed",
        returncode=result.returncode,
    )
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
