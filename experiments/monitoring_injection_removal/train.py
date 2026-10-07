"""Fresh standard-prompt 4B training on the frozen census-retained rows."""

from __future__ import annotations

import time

from experiments.monitoring_injection_removal.prepare import (
    CONFIG,
    DATA,
    OUTPUT,
    ROOT,
    configuration,
    resolved_profile,
    verify_preparation,
    write_json,
)
from gleipnir.monitoring_campaign_training import run_training


def make_job(config: dict, recipe: dict) -> dict:
    output = OUTPUT / "4b/filtered"
    return {
        **recipe,
        "job_name": "4b-census-filtered",
        "model": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "seed": config["seed"],
        "student_rows": str(DATA / "train/student_rows.jsonl"),
        "soft_targets": str(DATA / "soft_targets.jsonl"),
        "selection_manifest": None,
        "num_train_epochs": 1.0,
        "max_steps": -1,
        "learning_rate": config["learning_rate"],
        "soft_loss_weight": 1.0,
        "direct_loss_weight": 0.0,
        "completion_loss_weight": 0.0,
        "startup_validation_reference": str(
            ROOT / config["startup_validation_reference"]
        ),
        "save_steps": 1000000,
        "gradient_checkpointing": False,
        "output_dir": str(output),
        "causal_adapter_dir": str(output / "causal_adapter"),
        "model_dir": str(output / "model"),
        "hydra_log_dir": str(ROOT / "logs/runpod/monitoring_injection_removal/hydra"),
    }


def main() -> None:
    verify_preparation()
    config = configuration()
    run_training(
        ROOT,
        make_job(config, resolved_profile()["recipe"]),
        config,
        manifest_path=DATA / "manifest.json",
        config_path=CONFIG,
        source_paths=(
            "experiments/monitoring_injection_removal/train.py",
            "experiments/monitoring_injection_removal/prepare.py",
            "experiments/monitoring_hard_labels/train.py",
            "src/gleipnir/campaigns/training.py",
            "src/gleipnir/__init__.py",
            "src/gleipnir/_compat.py",
            "src/gleipnir/data/exclusions.py",
            "experiments/deception_distillation/train_student_sft.py",
            "src/gleipnir/campaigns/training_command.py",
            "src/gleipnir/training/packed.py",
            "src/gleipnir/packed_sequences.py",
        ),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json(
            OUTPUT / f"failure_train_{time.time_ns()}.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        raise
