"""Render the common selected-token monitoring training contract."""

from __future__ import annotations

import json
import sys
from typing import Any


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
        "student.soft_teacher_artifact="
        + (
            str(job["soft_targets"])
            if float(job.get("soft_loss_weight", 1.0))
            else "null"
        ),
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
    if implementation := job.get("attn_implementation"):
        command.append(f"student.attn_implementation={implementation}")
    for key in ("attention_backend_version", "attention_backend_canary_reference"):
        if value := job.get(key):
            command.append(f"student.training.{key}={value}")
    if "require_causal_conv1d" in job:
        required = str(bool(job["require_causal_conv1d"])).lower()
        command.append(f"student.training.require_causal_conv1d={required}")
    if "require_flash_sdpa" in job:
        required = str(bool(job["require_flash_sdpa"])).lower()
        command.append(f"student.training.require_flash_sdpa={required}")
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
    if "allow_unspec_int_on_nn_module" in job:
        enabled = str(bool(job["allow_unspec_int_on_nn_module"])).lower()
        command.append(f"student.training.allow_unspec_int_on_nn_module={enabled}")
    if "eager_attention_interface" in job:
        enabled = str(bool(job["eager_attention_interface"])).lower()
        command.append(f"student.training.eager_attention_interface={enabled}")
    for key in ("gated_delta_backend", "gated_delta_parity_policy"):
        if key in job:
            command.append(f"++student.training.{key}={job[key]}")
    if job.get("full_bf16_lora", False):
        command.extend(
            [
                "student.quantization.enabled=false",
                "++student.quantization.mlp_precision=bf16",
                "++student.quantization.full_bf16_lora=true",
            ]
        )
    if job.get("fp4_mlp_lora", False):
        command.extend(
            [
                "student.quantization.enabled=false",
                "++student.quantization.mlp_precision=fouroversix",
                "++student.quantization.full_bf16_lora=false",
                "++student.quantization.fp4_mlp_lora=true",
            ]
        )
        for key in (
            "fp4_backward_mode",
            "fp4_row_scaled_activations",
            "fp4_fused_row_scaling",
        ):
            value = job[key]
            encoded = str(value).lower() if isinstance(value, bool) else str(value)
            command.append(f"++student.quantization.{key}={encoded}")
    for key in (
        "sequence_packing",
        "packing_compile_cache_limit",
        "packed_attention_backend",
        "packed_attention_version",
        "packing_learning_gradient_tolerance",
        "packing_timing_authority",
        "expected_initial_master_sha256",
        "native_fp4_mlp",
        "native_fp4_mlp_parity_policy",
        "systems_adapter_scratch",
    ):
        if key in job:
            value = job[key]
            encoded = str(value).lower() if isinstance(value, bool) else str(value)
            command.append(f"++student.training.{key}={encoded}")
    if adaptive := job.get("adaptive_microbatching"):
        for key in ("enabled", "max_padded_tokens", "max_micro_batch_size", "profile"):
            if key in adaptive:
                value = adaptive[key]
                encoded = (
                    str(value).lower() if isinstance(value, bool) else str(int(value))
                )
                command.append(
                    f"student.training.adaptive_microbatching.{key}={encoded}"
                )
    if "selective_torch_compile_canary_tokens" in job:
        command.append(
            "student.training.selective_torch_compile_canary_tokens="
            f"{int(job['selective_torch_compile_canary_tokens'])}"
        )
    if trainer_optim := job.get("trainer_optim"):
        command.append(f"student.training.optim={trainer_optim}")
    for key in ("mil_loss_weight", "mil_temperature", "prefix_loss_weight"):
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
    if hydra_log_dir := job.get("hydra_log_dir"):
        command.append(f"hydra.run.dir={hydra_log_dir}/{job['job_name']}")
    selection = job.get("selection_manifest")
    command.append(
        "student.selection_manifest=null"
        if selection is None
        else f"student.selection_manifest={selection}"
    )
    if reference := job.get("startup_validation_reference"):
        command.append(f"++student.training.startup_validation_reference={reference}")
        if digest := job.get("startup_validation_reference_sha256"):
            command.append(
                f"++student.training.startup_validation_reference_sha256={digest}"
            )
    if job.get("concept_ablation_path"):
        command.extend(
            [
                f"++student.training.concept_ablation_path={job['concept_ablation_path']}",
                f"++student.training.concept_ablation_sha256={job['concept_ablation_sha256']}",
            ]
        )
    if "decision_tokens" in job:
        command.append(
            "++student.training.decision_tokens=" + json.dumps(job["decision_tokens"])
        )
    if "decision_prefix" in job:
        command.append(
            "++student.training.decision_prefix=" + json.dumps(job["decision_prefix"])
        )
    if job.get("per_record_binary_task"):
        command.append("++student.training.per_record_binary_task=true")
        mixture = job["task_mixture"]
        for key in ("monitoring_per_batch", "preference_per_batch"):
            command.append(f"++student.training.task_mixture.{key}={int(mixture[key])}")
        for key, weight in mixture["condition_weights"].items():
            command.append(
                f"++student.training.task_mixture.condition_weights.{key}={float(weight)}"
            )
    if "nonreentrant_checkpointing" in job:
        command.append(
            "++student.training.nonreentrant_checkpointing="
            + str(bool(job["nonreentrant_checkpointing"])).lower()
        )
    world_size = int(job.get("world_size", 1))
    if world_size not in {1, 2}:
        raise ValueError("validated launcher world size must be 1 or 2")
    if world_size > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world_size}",
            *command[1:],
        ]
    return command
