#!/usr/bin/env python3
"""Run one prepared data-scaling training job."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_job(path: Path, index: int) -> dict[str, Any]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not 0 <= index < len(lines):
        raise IndexError(f"job index {index} outside [0, {len(lines)})")
    return json.loads(lines[index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs", type=Path, default=Path("results/data_scaling/jobs.jsonl")
    )
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    args = parser.parse_args()

    jobs_path = args.jobs.resolve()
    job = load_job(jobs_path, args.index)
    selection = Path(job["selection_manifest"])
    if sha256_file(selection) != job["selection_sha256"]:
        raise ValueError(f"selection manifest checksum changed: {selection}")
    adapter_dir = Path(job["adapter_dir"])
    weights = adapter_dir / "adapter_model.safetensors"
    if weights.is_file() and not args.force:
        print(f"completed adapter already exists; skipping {job['job_name']}")
        return

    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "job.json").write_text(
        json.dumps(job, indent=2, sort_keys=True) + "\n"
    )
    command = [
        sys.executable,
        "experiments/deception_distillation/train_student_sft.py",
        "--config-path",
        "../data_scaling",
        "--config-name",
        "config",
        f"method=data_scaling_{job['job_name']}",
        f"output_dir={output_dir}",
        f"seed={job['seed']}",
        f"teacher.artifact={job['student_rows']}",
        f"student.soft_teacher_artifact={job['soft_targets']}",
        f"student.selection_manifest={job['selection_manifest']}",
        f"student.output_dir={adapter_dir}",
    ]
    if args.per_device_train_batch_size is not None:
        command.append(
            "student.training.per_device_train_batch_size="
            f"{args.per_device_train_batch_size}"
        )
    if args.gradient_accumulation_steps is not None:
        command.append(
            "student.training.gradient_accumulation_steps="
            f"{args.gradient_accumulation_steps}"
        )
    print("running", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not weights.is_file():
        raise RuntimeError(f"training returned without adapter weights: {weights}")


if __name__ == "__main__":
    main()
