#!/usr/bin/env python3
"""Run one prepared adapter-capacity training job."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gleipnir.qwen35_adapter_rebase import (  # noqa: E402
    MANIFEST_NAME,
    rebase_adapter,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def find_job(path: Path, job_name: str) -> dict[str, Any]:
    selected = [job for job in read_jsonl(path) if job["job_name"] == job_name]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one job named {job_name!r}")
    return selected[0]


def completed_model(job: dict[str, Any]) -> bool:
    model_dir = Path(job["model_dir"])
    metadata = model_dir / "training_metadata.json"
    if not metadata.is_file():
        return False
    if job["capacity_kind"] == "lora":
        causal_dir = Path(job["causal_adapter_dir"])
        return (
            (causal_dir / "adapter_model.safetensors").is_file()
            and (causal_dir / "training_metadata.json").is_file()
            and (model_dir / "adapter_model.safetensors").is_file()
            and (model_dir / MANIFEST_NAME).is_file()
        )
    return (model_dir / "model.safetensors").is_file() or (
        model_dir / "model.safetensors.index.json"
    ).is_file()


def training_command(
    job: dict[str, Any],
    *,
    distributed_processes: int,
) -> list[str]:
    full = job["capacity_kind"] == "full"
    command = []
    if full:
        if distributed_processes < 2:
            raise ValueError("full fine-tuning requires at least two processes")
        command.extend(
            [
                sys.executable,
                "-m",
                "accelerate.commands.launch",
                "--multi_gpu",
                "--num_processes",
                str(distributed_processes),
            ]
        )
    else:
        command.append(sys.executable)
    command.extend(
        [
            "experiments/deception_distillation/train_student_sft.py",
            "--config-path",
            "../adapter_capacity_scaling",
            "--config-name",
            "config",
            f"method=adapter_capacity_scaling_{job['job_name']}",
            f"output_dir={job['output_dir']}",
            f"seed={job['seed']}",
            f"teacher.artifact={job['student_rows']}",
            f"student.soft_teacher_artifact={job['soft_targets']}",
            "student.output_dir="
            + (job["model_dir"] if full else job["causal_adapter_dir"]),
            f"student.model_loader={'image_text_to_text' if full else 'causal_lm'}",
            f"student.finetuning_mode={'full' if full else 'lora'}",
            f"student.quantization.enabled={'false' if full else 'true'}",
            f"student.training.fsdp.enabled={'true' if full else 'false'}",
            "student.training.per_device_train_batch_size="
            + (
                "1"
                if full
                else str(job.get("micro_batch_size", 8))
            ),
            "student.training.gradient_accumulation_steps="
            + (
                "16"
                if full
                else str(job.get("gradient_accumulation_steps", 4))
            ),
        ]
    )
    if not full:
        command.extend(
            [
                f"student.lora.r={job['rank']}",
                f"student.lora.alpha={job['lora_alpha']}",
            ]
        )
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("results/adapter_capacity_scaling_qlora/lambda_jobs.jsonl"),
    )
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--distributed-processes", type=int, default=2)
    args = parser.parse_args()

    jobs_path = args.jobs.resolve()
    job = find_job(jobs_path, args.job_name)
    if completed_model(job):
        print(f"completed model already exists; skipping {job['job_name']}")
        return
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "job.json").write_text(
        json.dumps(job, indent=2, sort_keys=True) + "\n"
    )
    causal_dir = (
        Path(job["causal_adapter_dir"])
        if job["capacity_kind"] == "lora"
        else None
    )
    causal_complete = causal_dir is not None and (
        (causal_dir / "adapter_model.safetensors").is_file()
        and (causal_dir / "training_metadata.json").is_file()
    )
    if not causal_complete:
        command = training_command(
            job, distributed_processes=args.distributed_processes
        )
        print("running", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    if causal_dir is not None:
        manifest = rebase_adapter(causal_dir, Path(job["model_dir"]))
        print(
            f"rebased {job['job_name']} source={manifest['source_sha256']} "
            f"destination={manifest['destination_sha256']}",
            flush=True,
        )
    if not completed_model(job):
        raise RuntimeError(
            f"training returned without a complete model: {job['job_name']}"
        )


if __name__ == "__main__":
    main()
