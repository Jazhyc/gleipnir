"""Run the validated checkpointed 9B recipe on the identical frozen views."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from experiments.monitor_injection_augmentation_9b.prepare import (
    CONFIG,
    OUTPUT,
    ROOT,
    SHARED,
    configuration,
    verify_preparation,
)
from gleipnir.evaluation.sources import SCORING_SOURCE_PATHS
from gleipnir.monitoring_campaign_data import file_hash, write_json
from gleipnir.monitoring_campaign_runtime import training_environment
from gleipnir.monitoring_training_command import training_command
from gleipnir.qwen35_adapter_rebase import rebase_adapter


def make_job(config: dict, prior: dict) -> dict:
    """Change only data/artifact paths and attach the same-model validation receipt."""
    output = OUTPUT / "9b/augmented"
    job = {
        **prior,
        "job_name": "9b-monitor-injection-augmentation",
        "student_rows": str(SHARED / "student_rows.jsonl"),
        "soft_targets": str(
            ROOT / "data/student_injection_awareness/soft_targets.jsonl"
        ),
        "output_dir": str(output),
        "causal_adapter_dir": str(output / "causal_adapter"),
        "model_dir": str(output / "model"),
        "hydra_log_dir": str(
            ROOT / "logs/runpod/monitor_injection_augmentation_9b/hydra"
        ),
        "startup_validation_reference": str(
            ROOT / config["startup_validation_reference"]
        ),
    }
    if (
        job["model"] != config["model"]["id"]
        or job["model_revision"] != config["model"]["revision"]
        or job["learning_rate"] != config["learning_rate"]
        or job["learning_rate"] != 5e-5
        or job["seed"] != 0
        or job["num_train_epochs"] != 1
        or not job["gradient_checkpointing"]
        or not job["nonreentrant_checkpointing"]
        or job["selective_torch_compile_policy"]
        != "checkpointed_full_attention_and_linear_shell"
        or job["effective_batch_size"] != 32
        or job["rank"] != 128
        or job["lora_alpha"] != 256
        or job["adaptive_microbatching"]["max_padded_tokens"] != 16384
        or job["soft_loss_weight"] != 1
        or job["direct_loss_weight"] != 0
    ):
        raise ValueError("frozen 9B recipe drift")
    return job


def validate_metadata(metadata: dict, config: dict) -> None:
    """Compare recipe metadata to the prior successful 9B condition."""
    reference = json.loads((ROOT / config["startup_validation_reference"]).read_text())
    for key in (
        "model",
        "model_revision",
        "seed",
        "gradient_checkpointing",
        "gradient_checkpointing_policy",
        "gradient_checkpointing_layer_indices",
        "checkpointed_layer_indices",
        "sequence_lengths",
        "lora_trainable_parameters",
    ):
        if metadata[key] != reference[key]:
            raise ValueError(f"9B metadata drift: {key}")
    for key in (
        "max_packed_tokens",
        "enabled",
        "bf16_matmul",
        "compile_cache_limit",
        "emulate_precision_casts",
        "fail_on_recompile_limit_hit",
    ):
        if metadata["sequence_packing"][key] != reference["sequence_packing"][key]:
            raise ValueError(f"9B packing recipe drift: {key}")
    if (
        metadata["training_state"]["global_step"] != 272
        or metadata["sequence_packing"]["initial_master_sha256"]
        != config["model"]["initial_master_sha256"]
        or metadata["sequence_packing"]["initial_master_sha256"]
        == metadata["sequence_packing"]["final_master_sha256"]
        or metadata["startup_validation"]["reference_sha256"]
        != config["startup_validation_reference_sha256"]
        or metadata["startup_validation"]["performed_this_run"]
        or metadata["quantization"]["enabled"]
        or not metadata["quantization"]["full_bf16_lora"]["verified"]
        or metadata["quantization"]["full_bf16_lora"]["master_dtype"] != "torch.float32"
        or metadata["quantization"]["full_bf16_lora"]["frozen_dtype"]
        != "torch.bfloat16"
        or metadata["gated_delta_backend"]["backend"] != "flashqla"
        or metadata["gated_delta_backend"]["replaced_layers"] != 24
        or metadata["selective_torch_compile"]["policy"]
        != "checkpointed_full_attention_and_linear_shell"
        or metadata["optimization"]["learning_rate"] != 5e-5
        or metadata["losses"]["soft_weight"] != 1
        or metadata["losses"]["direct_weight"] != 0
        or metadata["losses"]["completion_weight"] != 0
        or metadata["losses"]["accumulation_policy"]
        != "sum_per_example_over_logical_batch_v1"
        or not metadata["adaptive_microbatching"]["require_finite_gradients"]
    ):
        raise ValueError("completed 9B initialization/recipe/objective/reuse drift")


def main() -> None:
    config = configuration()
    verify_preparation()
    job = make_job(config, json.loads((ROOT / config["prior_job"]).read_text()))
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "job.json").exists() and json.loads(
        (output / "job.json").read_text()
    ) != job:
        raise ValueError("existing 9B job drift")
    write_json(output / "job.json", job)
    master = Path(job["causal_adapter_dir"])
    if not (master / "training_metadata.json").exists():
        audit_path = (
            ROOT
            / "results/monitor_injection_augmentation/4b/augmented/token_audit.json"
        )
        audit = json.loads(audit_path.read_text())
        if (
            audit["rows"] != 8688
            or audit["maximum"] > job["max_length"]
            or audit["truncated"]
        ):
            raise ValueError("reused identical-tokenizer input audit failed")
        write_json(
            output / "token_audit.json",
            {
                **audit,
                "performed_this_run": False,
                "policy": "identical_materialized_rows_and_tokenizer_files",
                "reference_sha256": file_hash(audit_path),
            },
        )
        command = training_command(job)
        sources = (
            "experiments/monitor_injection_augmentation_9b/train.py",
            "experiments/monitor_injection_augmentation_9b/prepare.py",
            "experiments/monitor_injection_augmentation_9b/evaluate.py",
            "experiments/monitor_injection_augmentation_9b/launch_evaluate.py",
            "src/gleipnir/validated_startup.py",
            "src/gleipnir/monitoring_campaign_runtime.py",
            "src/gleipnir/campaigns/training_command.py",
            *SCORING_SOURCE_PATHS,
            "src/gleipnir/training/packed.py",
            "src/gleipnir/packed_sequences.py",
            "experiments/deception_distillation/train_student_sft.py",
            "experiments/monitor_injection_augmentation/metrics.py",
        )
        write_json(
            output / "execution_contract.json",
            {
                "job": job,
                "command": command,
                "manifest_sha256": file_hash(
                    ROOT / "data/monitor_injection_augmentation_9b/manifest.json"
                ),
                "source_sha256": {s: file_hash(ROOT / s) for s in sources},
            },
        )
        write_json(
            output / "status.json",
            {"stage": "training", "validation": "reuse_recorded_9b_recipe"},
        )
        subprocess.run(
            command,
            cwd=ROOT,
            env=training_environment(ROOT, config["campaign_id"]),
            check=True,
        )
    metadata = json.loads((master / "training_metadata.json").read_text())
    validate_metadata(metadata, config)
    rebase_adapter(master, Path(job["model_dir"]))
    write_json(
        output / "complete.json",
        {
            "status": "trained",
            "steps": 272,
            "training_rows": 8688,
            "config_sha256": file_hash(CONFIG),
            "manifest_sha256": file_hash(
                ROOT / "data/monitor_injection_augmentation_9b/manifest.json"
            ),
            "master_sha256": file_hash(master / "adapter_model.safetensors"),
            "serving_sha256": file_hash(
                Path(job["model_dir"]) / "adapter_model.safetensors"
            ),
            "teacher_targets_sha256": file_hash(Path(job["soft_targets"])),
        },
    )
    write_json(output / "status.json", {"stage": "trained", "steps": 272})
    print("training_complete monitor_injection_augmentation_9b", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json(
            OUTPUT / f"failure_train_{time.time_ns()}.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        raise
