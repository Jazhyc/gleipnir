"""Train the matched fresh adapter on the frozen 40% replacement population."""

from __future__ import annotations

import math
import time

import yaml

from experiments.monitor_injection_augmentation.prepare import (
    CONFIG,
    DATA,
    OUTPUT,
    ROOT,
    verify_preparation,
)
from experiments.monitoring_hard_labels.prepare import resolved_profile
from gleipnir.evaluation.sources import SCORING_SOURCE_PATHS
from gleipnir.monitoring_campaign_data import write_json
from gleipnir.monitoring_campaign_training import run_training


def make_job(config: dict, recipe: dict) -> dict:
    """Preserve the original duration, initialization and soft BCE objective."""
    output = OUTPUT / "4b/augmented"
    return {
        **recipe,
        "job_name": "4b-monitor-injection-augmentation",
        "model": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "seed": config["seed"],
        "student_rows": str(DATA / "student_rows.jsonl"),
        "soft_targets": str(ROOT / config["teacher_source"]),
        "selection_manifest": None,
        "train_rows": config["training_rows"],
        "num_train_epochs": config["epochs"],
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
        "hydra_log_dir": str(ROOT / "logs/runpod/monitor_injection_augmentation/hydra"),
    }


def main() -> None:
    manifest = verify_preparation()
    config = yaml.safe_load(CONFIG.read_text())
    config["expected_steps"] = math.ceil(
        config["training_rows"] / config["logical_batch_size"]
    )
    if config["expected_steps"] != manifest["summary"]["expected_steps"]:
        raise ValueError("matched training duration drift")
    from safetensors import safe_open

    fresh = ROOT / config["model"]["initial_adapter"]
    with safe_open(str(fresh / "adapter_model.safetensors"), framework="pt") as handle:
        b_keys = [k for k in handle.keys() if "lora_B" in k]
        if len(b_keys) != 128 or any(
            handle.get_tensor(k).count_nonzero() for k in b_keys
        ):
            raise ValueError("initial adapter is not the fresh zero-B rank-128 layout")
    run_training(
        ROOT,
        make_job(config, resolved_profile()["recipe"]),
        config,
        manifest_path=DATA / "manifest.json",
        config_path=CONFIG,
        source_paths=(
            "experiments/monitor_injection_augmentation/train.py",
            "experiments/monitor_injection_augmentation/prepare.py",
            "experiments/monitor_injection_augmentation/prepare_eval.py",
            "experiments/monitor_injection_augmentation/evaluate.py",
            "experiments/monitor_injection_augmentation/metrics.py",
            "experiments/monitor_injection_augmentation/eval_config.yaml",
            "experiments/monitor_injection_augmentation/launch_evaluate.py",
            *SCORING_SOURCE_PATHS,
            "experiments/monitoring_hard_labels/train.py",
            "src/gleipnir/monitoring_campaign_training.py",
            "src/gleipnir/monitoring_campaign_runtime.py",
            "experiments/deception_distillation/train_student_sft.py",
            "src/gleipnir/monitoring_training_command.py",
            "src/gleipnir/packed_training.py",
            "src/gleipnir/packed_sequences.py",
        ),
    )
    print("training_complete monitor_injection_augmentation", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json(
            OUTPUT / f"failure_train_{time.time_ns()}.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise
