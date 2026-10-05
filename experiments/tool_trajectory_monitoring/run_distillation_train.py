#!/usr/bin/env python3
"""Train and checksum-rebase one prepared Kimi-soft scaling adapter."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from experiments.tool_trajectory_monitoring.distillation_scaling import validate_jobs
from gleipnir.monitoring_training_command import training_command
from gleipnir.qwen35_adapter_rebase import MANIFEST_NAME, rebase_adapter

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def find_job(
    path: Path, job_name: str, *, validate_design: bool = True
) -> dict[str, Any]:
    jobs = read_jsonl(path)
    if validate_design:
        validate_jobs(jobs)
    selected = [job for job in jobs if job["job_name"] == job_name]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one job named {job_name!r}")
    return selected[0]


def completed_model(job: dict[str, Any]) -> bool:
    causal_dir = Path(job["causal_adapter_dir"])
    model_dir = Path(job["model_dir"])
    return (
        (causal_dir / "adapter_model.safetensors").is_file()
        and (causal_dir / "training_metadata.json").is_file()
        and (model_dir / "adapter_model.safetensors").is_file()
        and (model_dir / MANIFEST_NAME).is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("results/tool_trajectory_distillation_scaling/lambda_jobs.jsonl"),
    )
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--allow-preflight-job", action="store_true")
    parser.add_argument("--allow-non-scaling-job", action="store_true")
    args = parser.parse_args()

    job = find_job(
        args.jobs.resolve(),
        args.job_name,
        validate_design=not (args.allow_preflight_job or args.allow_non_scaling_job),
    )
    if completed_model(job):
        print(f"completed model already exists; skipping {job['job_name']}")
        return
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "job.json").write_text(
        json.dumps(job, indent=2, sort_keys=True) + "\n"
    )
    causal_dir = Path(job["causal_adapter_dir"])
    causal_complete = (causal_dir / "adapter_model.safetensors").is_file() and (
        causal_dir / "training_metadata.json"
    ).is_file()
    if not causal_complete:
        command = training_command(job)
        print("running", " ".join(command), flush=True)
        environment = dict(os.environ)
        if job.get("gated_delta_backend", "fla") == "flashqla":
            from gleipnir.flashqla_training import flashqla_environment

            environment = flashqla_environment(environment)
        if job.get("packed_attention_backend") == "flash_attention_4":
            from gleipnir.attention_backends import packed_fa4_environment

            environment = packed_fa4_environment(environment, ROOT)
        if job.get("native_fp4_mlp", False):
            from gleipnir.native_fp4_training import native_fp4_environment

            environment = native_fp4_environment(environment, ROOT)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    manifest = rebase_adapter(causal_dir, Path(job["model_dir"]))
    print(
        f"rebased {job['job_name']} source={manifest['source_sha256']} "
        f"destination={manifest['destination_sha256']}",
        flush=True,
    )
    if not completed_model(job):
        raise RuntimeError(
            f"training returned without complete model: {job['job_name']}"
        )


if __name__ == "__main__":
    main()
