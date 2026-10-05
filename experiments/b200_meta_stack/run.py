"""Isolated sequential GPU candidates using shared compatible compiler caches."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import yaml

from experiments.b200_nvidia_mxfp8.run import environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "baseline",
            "persistent_dq",
            "persistent_dkdv",
            "persistent_both",
            "ds_warp_amax",
            "persistent_ds_warp_amax",
            "dq_store128",
        ],
    )
    args = parser.parse_args()
    if not args.attempt.isalnum():
        raise ValueError("attempt must be alphanumeric")
    cfg = yaml.safe_load(Path("experiments/b200_meta_stack/config.yaml").read_text())
    root = Path("results/b200_meta_stack") / args.attempt
    root.mkdir(parents=True, exist_ok=False)
    logs = Path(cfg["logs"]) / args.attempt
    logs.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, **environment(cfg)}
    report = {"status": "running", "config": cfg, "candidates": []}
    for variant in args.variants:
        if variant in {"producer", "producer128"}:
            command = [
                sys.executable,
                "-m",
                "experiments.b200_meta_stack.producer_probe",
                "--output",
                str(root / variant),
            ]
            if variant == "producer128":
                command += ["--dq-store-bits", "128"]
        elif variant in {"projection_bf16", "projection_mxfp8"}:
            command = [
                sys.executable,
                "-m",
                "experiments.b200_meta_stack.projection_probe",
                "--output",
                str(root / variant),
            ]
            if variant.endswith("mxfp8"):
                command.append("--mxfp8")
        else:
            command = [
                sys.executable,
                "-m",
                "experiments.b200_meta_stack.native_probe",
                "--variant",
                variant,
                "--output",
                str(root / variant),
            ]
        row = {"variant": variant, "command": command, "status": "starting"}
        report["candidates"].append(row)
        (root / "launch.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"starting {variant}", flush=True)
        with (logs / (variant + ".log")).open("x") as handle:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=cfg["timeout_seconds"])
                row.update(
                    status="complete" if code == 0 else "failed", returncode=code
                )
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                row.update(status="timeout", returncode=process.returncode)
        (root / "launch.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    report["status"] = "complete"
    (root / "launch.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
