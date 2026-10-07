"""Train a fresh monitoring-anchored adapter with the selected packed recipe."""

from __future__ import annotations

import json
import subprocess
import time

from experiments.judge_injection_joint.prepare import (
    CONFIG,
    DATA,
    OUTPUT,
    ROOT,
    configuration,
    verify_preparation,
)
from experiments.monitoring_hard_labels.prepare import resolved_profile
from experiments.monitoring_hard_labels.train import validate_training_metadata
from gleipnir.evaluation.sources import DECISION_SOURCE_PATHS
from gleipnir.monitoring_campaign_data import file_hash, write_json
from gleipnir.monitoring_campaign_runtime import training_environment
from gleipnir.monitoring_training_command import training_command
from gleipnir.qwen35_adapter_rebase import rebase_adapter


def make_job(config: dict, manifest: dict) -> dict:
    output = OUTPUT / "4b/joint"
    return {
        **resolved_profile()["recipe"],
        "job_name": "judge-injection-joint-4b",
        "model": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "seed": config["seed"],
        "student_rows": str(DATA / "student_rows.jsonl"),
        "soft_targets": str(ROOT / config["teacher_source"]),
        "selection_manifest": None,
        "num_train_epochs": config["epochs"],
        "max_steps": -1,
        "train_rows": manifest["training_draws"],
        "learning_rate": config["learning_rate"],
        "soft_loss_weight": 1.0,
        "direct_loss_weight": 1.0,
        "completion_loss_weight": 0.0,
        "per_record_binary_task": True,
        "task_mixture": config["task_mixture"],
        "save_steps": 1000000,
        "gradient_checkpointing": False,
        "startup_validation_reference": str(
            ROOT / config["startup_validation_reference"]
        ),
        "output_dir": str(output),
        "causal_adapter_dir": str(output / "causal_adapter"),
        "model_dir": str(output / "model"),
        "hydra_log_dir": str(ROOT / "logs/runpod/judge_injection_joint/hydra"),
    }


def main() -> None:
    config, manifest = configuration(), verify_preparation()
    job = make_job(config, manifest)
    output = OUTPUT / "4b/joint"
    master = output / "causal_adapter"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "job.json").exists() and json.loads(
        (output / "job.json").read_text()
    ) != job:
        raise ValueError("joint training job drift")
    write_json(output / "job.json", job)
    if not (master / "training_metadata.json").exists():
        command = training_command(job)
        command.append(
            f"student.init_adapter={ROOT / config['model']['fresh_adapter']}"
        )
        source_paths = [
            "experiments/judge_injection_joint/train.py",
            "experiments/judge_injection_joint/prepare.py",
            "experiments/deception_distillation/train_student_sft.py",
            "src/gleipnir/training/binary_tasks.py",
            *DECISION_SOURCE_PATHS,
            "src/gleipnir/campaigns/training_command.py",
            "src/gleipnir/training/packing.py",
            "src/gleipnir/training/packed.py",
        ]
        write_json(
            output / "execution_contract.json",
            {
                "job": job,
                "command": command,
                "manifest_sha256": file_hash(DATA / "manifest.json"),
                "source_sha256": {p: file_hash(ROOT / p) for p in source_paths},
                "initial_adapter": config["model"]["fresh_adapter"],
                "initial_adapter_kind": (
                    "untrained; all 128 lora_B matrices verified zero"
                ),
                "optimizer": "fresh AdamW; fresh adapter; frozen BF16 base",
            },
        )
        write_json(
            output / "status.json",
            {
                "stage": "training",
                "expected_steps": manifest["expected_steps"],
                "validation": "reuse_recorded_recipe",
            },
        )
        subprocess.run(
            command,
            cwd=ROOT,
            env=training_environment(ROOT, config["campaign_id"]),
            check=True,
        )
    metadata = json.loads((master / "training_metadata.json").read_text())
    reference = json.loads((ROOT / config["startup_validation_reference"]).read_text())
    validate_training_metadata(
        metadata,
        job,
        reference["sequence_packing"]["initial_master_sha256"],
        expected_steps=manifest["expected_steps"],
    )
    if (
        metadata["optimization"]["task_mixture"] != manifest["sampler"]
        or not metadata["losses"]["per_record_binary_task"]
    ):
        raise ValueError("actual sampling/objective contract drift")
    rebase_adapter(master, output / "model")
    write_json(
        output / "complete.json",
        {
            "status": "trained",
            "steps": manifest["expected_steps"],
            "config_sha256": file_hash(CONFIG),
            "manifest_sha256": file_hash(DATA / "manifest.json"),
            "master_sha256": file_hash(master / "adapter_model.safetensors"),
            "serving_sha256": file_hash(output / "model/adapter_model.safetensors"),
            "sampler": manifest["sampler"],
        },
    )
    write_json(
        output / "status.json",
        {"stage": "trained", "steps": manifest["expected_steps"]},
    )
    print("training_complete judge_injection_joint", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json(
            OUTPUT / f"failure_train_{time.time_ns()}.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        raise
