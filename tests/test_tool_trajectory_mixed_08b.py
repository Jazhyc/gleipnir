import json
from pathlib import Path

import pytest

from experiments.tool_trajectory_monitoring.prepare_mixed_08b_distillation import (
    GRADIENT_ACCUMULATION_STEPS,
    JOB_NAME,
    MICRO_BATCH_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    make_job,
    validate_mixed_08b_jobs,
)
from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)
from experiments.tool_trajectory_monitoring.run_mixed_08b_lambda import recipe


def test_mixed_08b_job_is_one_epoch_rank128_soft_only(tmp_path: Path) -> None:
    job = make_job(tmp_path / "data", tmp_path / "results")
    validate_mixed_08b_jobs([job])

    assert job["job_name"] == JOB_NAME
    assert job["model"] == MODEL_ID
    assert job["model_revision"] == MODEL_REVISION
    assert job["train_rows"] == 21_837
    assert job["rank"] == 128
    assert job["num_train_epochs"] == 1.0
    assert job["max_steps"] == -1
    assert job["micro_batch_size"] == MICRO_BATCH_SIZE == 8
    assert job["gradient_accumulation_steps"] == GRADIENT_ACCUMULATION_STEPS == 4
    assert job["effective_batch_size"] == 32

    command = training_command(job)
    assert "student.training.soft_loss_weight=1.0" in command
    assert "student.training.direct_loss_weight=0.0" in command
    assert "student.training.completion_loss_weight=0.0" in command
    assert "student.training.per_device_train_batch_size=8" in command
    assert "student.training.gradient_accumulation_steps=4" in command


def test_mixed_08b_validation_fails_on_recipe_drift(tmp_path: Path) -> None:
    job = make_job(tmp_path / "data", tmp_path / "results")
    job["direct_loss_weight"] = 1.0
    with pytest.raises(ValueError, match="direct_loss_weight"):
        validate_mixed_08b_jobs([job])


def test_mixed_08b_runtime_recipe_is_explicit(tmp_path: Path) -> None:
    frozen = json.loads(json.dumps(recipe()))
    assert frozen == {
        "target": "kimi_soft_only",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "rank": 128,
        "micro_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 32,
        "max_length": 29_696,
        "quantization": "nf4-double-quant-bf16",
        "direct_logits_mode": "selected_positions",
        "epochs": 1,
    }
