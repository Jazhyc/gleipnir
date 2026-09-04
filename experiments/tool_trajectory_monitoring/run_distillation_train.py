#!/usr/bin/env python3
"""Train and checksum-rebase one prepared Kimi-soft scaling adapter."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments.tool_trajectory_monitoring.distillation_scaling import validate_jobs
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


def training_command(job: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "experiments/deception_distillation/train_student_sft.py",
        "--config-path",
        "../tool_trajectory_monitoring",
        "--config-name",
        "distillation_config",
        f"method={job['job_name']}",
        f"output_dir={job['output_dir']}",
        f"seed={job['seed']}",
        f"teacher.artifact={job['student_rows']}",
        f"student.soft_teacher_artifact={job['soft_targets']}",
        f"student.output_dir={job['causal_adapter_dir']}",
        f"student.max_length={job['max_length']}",
        f"student.lora.r={job['rank']}",
        f"student.lora.alpha={job['lora_alpha']}",
        f"student.training.soft_loss_weight={float(job.get('soft_loss_weight', 1.0))}",
        "student.training.direct_loss_weight="
        f"{float(job.get('direct_loss_weight', 0.0))}",
        "student.training.completion_loss_weight="
        f"{float(job.get('completion_loss_weight', 0.0))}",
        f"student.training.learning_rate={job['learning_rate']}",
        f"student.training.num_train_epochs={job['num_train_epochs']}",
        f"student.training.max_steps={job['max_steps']}",
        f"student.training.per_device_train_batch_size={job['micro_batch_size']}",
        "student.training.gradient_accumulation_steps="
        f"{job['gradient_accumulation_steps']}",
        f"student.training.save_steps={job['save_steps']}",
    ]
    if "completion_max_length" in job:
        command.append(
            f"student.completion_max_length={int(job['completion_max_length'])}"
        )
    if "require_causal_conv1d" in job:
        required = str(bool(job["require_causal_conv1d"])).lower()
        command.append(f"student.training.require_causal_conv1d={required}")
    if sampling_strategy := job.get("train_sampling_strategy"):
        command.append(f"student.training.train_sampling_strategy={sampling_strategy}")
    if "gradient_checkpointing" in job:
        enabled = str(bool(job["gradient_checkpointing"])).lower()
        command.append(f"student.training.gradient_checkpointing={enabled}")
    if checkpointing_policy := job.get("gradient_checkpointing_policy"):
        command.append(
            f"student.training.gradient_checkpointing_policy={checkpointing_policy}"
        )
    checkpointing_indices = job.get("gradient_checkpointing_layer_indices")
    if checkpointing_indices is not None:
        encoded_indices = ",".join(str(int(index)) for index in checkpointing_indices)
        command.append(
            f"student.training.gradient_checkpointing_layer_indices=[{encoded_indices}]"
        )
    if compile_policy := job.get("selective_torch_compile_policy"):
        command.append(
            f"student.training.selective_torch_compile_policy={compile_policy}"
        )
    if compile_backend := job.get("selective_torch_compile_backend"):
        command.append(
            f"student.training.selective_torch_compile_backend={compile_backend}"
        )
    if compile_mode := job.get("selective_torch_compile_mode"):
        command.append(f"student.training.selective_torch_compile_mode={compile_mode}")
    if "selective_torch_compile_dynamic" in job:
        dynamic = str(bool(job["selective_torch_compile_dynamic"])).lower()
        command.append(f"student.training.selective_torch_compile_dynamic={dynamic}")
    if "selective_torch_compile_canary_tokens" in job:
        command.append(
            "student.training.selective_torch_compile_canary_tokens="
            f"{int(job['selective_torch_compile_canary_tokens'])}"
        )
    if trainer_optim := job.get("trainer_optim"):
        command.append(f"student.training.optim={trainer_optim}")
    for key in ("mil_loss_weight", "mil_temperature"):
        if key in job:
            command.append(f"student.training.{key}={float(job[key])}")
    for key in ("mil_top_k", "mil_max_instances"):
        if key in job:
            command.append(f"student.training.{key}={int(job[key])}")
    if mil_pooling := job.get("mil_pooling"):
        command.append(f"student.training.mil_pooling={mil_pooling}")
    if completion_logits_mode := job.get("completion_logits_mode"):
        command.append(
            f"student.training.completion_logits_mode={completion_logits_mode}"
        )
    if "completion_projection_chunk_size" in job:
        command.append(
            "student.training.completion_projection_chunk_size="
            f"{int(job['completion_projection_chunk_size'])}"
        )
    if "sequential_objective_backward" in job:
        sequential = str(bool(job["sequential_objective_backward"])).lower()
        command.append(f"student.training.sequential_objective_backward={sequential}")
    if model := job.get("model"):
        command.append(f"student.model={model}")
    if model_revision := job.get("model_revision"):
        command.append(f"student.model_revision={model_revision}")
    selection = job.get("selection_manifest")
    command.append(
        "student.selection_manifest=null"
        if selection is None
        else f"student.selection_manifest={selection}"
    )
    return command


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
        subprocess.run(command, cwd=ROOT, check=True)
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
