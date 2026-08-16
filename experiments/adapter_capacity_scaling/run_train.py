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
            str(job.get("config_path", "../adapter_capacity_scaling")),
            "--config-name",
            str(job.get("config_name", "config")),
            f"method=adapter_capacity_scaling_{job['job_name']}",
            f"output_dir={job['output_dir']}",
            f"seed={job['seed']}",
            f"teacher.artifact={job['student_rows']}",
            "student.output_dir="
            + (job["model_dir"] if full else job["causal_adapter_dir"]),
            f"student.model_loader={'image_text_to_text' if full else 'causal_lm'}",
            f"student.finetuning_mode={'full' if full else 'lora'}",
            "student.quantization.enabled="
            + (
                "false"
                if full
                else str(bool(job.get("quantization_enabled", True))).lower()
            ),
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
    if float(job.get("soft_loss_weight", 1.0)) > 0:
        command.append(f"student.soft_teacher_artifact={job['soft_targets']}")
    if not full:
        command.extend(
            [
                f"student.lora.r={job['rank']}",
                f"student.lora.alpha={job['lora_alpha']}",
            ]
        )
        if "lora_dropout" in job:
            command.append(f"student.lora.dropout={job['lora_dropout']}")
        if "lora_target_modules" in job:
            modules = ",".join(str(value) for value in job["lora_target_modules"])
            command.append(f"student.lora.target_modules=[{modules}]")
        if "lora_init" in job:
            command.append(f"student.lora.init={job['lora_init']}")
        if "lora_use_dora" in job:
            command.append(
                f"student.lora.use_dora={str(bool(job['lora_use_dora'])).lower()}"
            )
        for job_key, config_key in (
            ("eva_rho", "student.lora.eva_rho"),
            ("eva_tau", "student.lora.eva_tau"),
            ("eva_rows", "student.lora.eva_rows"),
        ):
            if job_key in job:
                command.append(f"{config_key}={job[job_key]}")
    selection_manifest = job.get("selection_manifest")
    if selection_manifest is not None:
        command.append(f"student.selection_manifest={selection_manifest}")
    optional_training_overrides = {
        "optimizer": "student.training.optimizer",
        "learning_rate": "student.training.learning_rate",
        "lr_scheduler_type": "student.training.lr_scheduler_type",
        "num_train_epochs": "student.training.num_train_epochs",
        "warmup_ratio": "student.training.warmup_ratio",
        "weight_decay": "student.training.weight_decay",
        "muon_adjust_lr_fn": "student.training.muon_adjust_lr_fn",
        "dataset_sampling": "student.training.dataset_sampling",
        "soft_target_logit_scale": "student.training.soft_target_logit_scale",
        "save_strategy": "student.training.save_strategy",
        "save_steps": "student.training.save_steps",
        "save_total_limit": "student.training.save_total_limit",
        "save_only_model": "student.training.save_only_model",
        "max_steps": "student.training.max_steps",
        "soft_loss_weight": "student.training.soft_loss_weight",
        "direct_loss_weight": "student.training.direct_loss_weight",
        "soft_loss_type": "student.training.soft_loss_type",
        "soft_huber_delta": "student.training.soft_huber_delta",
        "dataset_loss_weighting": "student.training.dataset_loss_weighting",
        "group_dro_eta": "student.training.group_dro_eta",
        "group_dro_ema": "student.training.group_dro_ema",
        "decision_head_mode": "student.training.decision_head_mode",
        "decision_head_init": "student.training.decision_head_init",
    }
    for job_key, config_key in optional_training_overrides.items():
        if job_key in job:
            command.append(f"{config_key}={job[job_key]}")
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
