"""Continue the regular FP32 LoRA with A/B preference cross-entropy."""

from __future__ import annotations

import json
import subprocess
import time

from experiments.judge_injection_continuation.prepare import (
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


def main() -> None:
    config = configuration()
    manifest = verify_preparation()
    output = OUTPUT / "4b/continued"
    master = output / "causal_adapter"
    job = {
        **resolved_profile()["recipe"],
        "job_name": "judge-injection-continuation-4b",
        "model": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "seed": config["seed"],
        "student_rows": str(DATA / "train.jsonl"),
        "soft_targets": None,
        "selection_manifest": None,
        "num_train_epochs": config["epochs"],
        "max_steps": -1,
        "train_rows": manifest["training_rows"],
        "learning_rate": config["learning_rate"],
        "soft_loss_weight": 0.0,
        "direct_loss_weight": 1.0,
        "completion_loss_weight": 0.0,
        "decision_tokens": ["A", "B"],
        "decision_prefix": "",
        "save_steps": 1000000,
        "gradient_checkpointing": False,
        "startup_validation_reference": str(
            ROOT / config["startup_validation_reference"]
        ),
        "output_dir": str(output),
        "causal_adapter_dir": str(master),
        "model_dir": str(output / "model"),
        "hydra_log_dir": str(ROOT / "logs/runpod/judge_injection_continuation/hydra"),
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "job.json").exists() and json.loads(
        (output / "job.json").read_text()
    ) != job:
        raise ValueError("training job drift")
    write_json(output / "job.json", job)
    if not (master / "training_metadata.json").exists():
        command = training_command(job)
        command.append(
            f"student.init_adapter={ROOT / config['model']['initial_adapter']}"
        )
        source_paths = [
            "experiments/judge_injection_continuation/train.py",
            "experiments/deception_distillation/train_student_sft.py",
            *DECISION_SOURCE_PATHS,
            "src/gleipnir/monitoring_training_command.py",
            "src/gleipnir/packed_training.py",
        ]
        write_json(
            output / "execution_contract.json",
            {
                "job": job,
                "command": command,
                "manifest_sha256": file_hash(DATA / "manifest.json"),
                "source_sha256": {p: file_hash(ROOT / p) for p in source_paths},
                "initial_master_file_sha256": manifest["source_artifact_sha256"][
                    config["model"]["initial_adapter"] + "/adapter_model.safetensors"
                ],
                "optimizer": (
                    "fresh AdamW; adapter weights continued; "
                    "original checkpoint preserved"
                ),
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
    original = json.loads((ROOT / config["startup_validation_reference"]).read_text())
    validate_training_metadata(
        metadata,
        job,
        original["sequence_packing"]["final_master_sha256"],
        expected_steps=manifest["expected_steps"],
    )
    if (
        metadata["losses"]["decision_token_ids"] != [32, 33]
        or metadata["losses"]["decision_prefix"] != ""
    ):
        raise ValueError("A/B decision surface drift")
    rebase_adapter(master, output / "model")
    write_json(
        output / "complete.json",
        {
            "status": "trained",
            "steps": manifest["expected_steps"],
            "training_rows": manifest["training_rows"],
            "config_sha256": file_hash(CONFIG),
            "manifest_sha256": file_hash(DATA / "manifest.json"),
            "master_sha256": file_hash(master / "adapter_model.safetensors"),
            "serving_sha256": file_hash(output / "model/adapter_model.safetensors"),
        },
    )
    write_json(
        output / "status.json",
        {"stage": "trained", "steps": manifest["expected_steps"]},
    )
    print("training_complete judge_injection_continuation", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json(
            OUTPUT / f"failure_train_{time.time_ns()}.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        raise
