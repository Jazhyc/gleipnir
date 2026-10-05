"""Require exact full-run controls, complete physical coverage and paired ID rows."""

from copy import deepcopy

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.b200_fp4_full_training.campaign import (
    ROOT,
    configuration,
    make_job,
    profile,
    validate_prediction_membership,
)
from experiments.b200_fp4_full_training.compile_ahead import future_shapes
from gleipnir.monitoring_training_command import training_command
from gleipnir.packed_training import validate_packed_training_config


def test_full_replication_uses_original_regular_controls_and_native_recipe():
    config = configuration()
    job = make_job(config, profile(config)["recipe"])
    assert job["train_rows"] == 8688
    assert config["expected_steps"] == 272
    assert job["learning_rate"] == 5e-5
    assert job["seed"] == 0
    assert job["num_train_epochs"] == 1.0 and job["max_steps"] == -1
    assert job["soft_loss_weight"] == 1.0 and job["direct_loss_weight"] == 0.0
    assert job["selection_manifest"] is None
    assert job["expected_initial_master_sha256"] == config["initial_master_sha256"]
    command = training_command(job)
    with initialize_config_dir(
        config_dir=str(ROOT / "experiments/tool_trajectory_monitoring"),
        version_base=None,
    ):
        student = OmegaConf.to_container(
            compose(config_name="distillation_config", overrides=command[6:]).student,
            resolve=True,
        )
    assert validate_packed_training_config(student)
    assert student["training"]["native_fp4_mlp"] is True
    assert "packing_timing_authority" not in student["training"]


def populations():
    inputs = [
        {
            "id": "a",
            "metadata": {
                "ground_truth": 0,
                "source_dataset": "test_stride",
                "rendered_prompt_sha256": "prompt-a",
            },
        },
        {
            "id": "b",
            "metadata": {
                "ground_truth": 1,
                "source_dataset": "gloom_exfiltration",
                "rendered_prompt_sha256": "prompt-b",
            },
        },
    ]
    predictions = [
        {
            "id": "a",
            "label": 0,
            "source": "test_stride",
            "source_prompt_sha256": "prompt-a",
            "score": 0.1,
        },
        {
            "id": "b",
            "label": 1,
            "source": "gloom_exfiltration",
            "source_prompt_sha256": "prompt-b",
            "score": 0.9,
        },
    ]
    return inputs, predictions


def test_id_matching_allows_order_changes_but_binds_every_label_prompt_and_source():
    inputs, predictions = populations()
    validate_prediction_membership(list(reversed(predictions)), inputs)


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "extra", "label", "source", "prompt", "nonfinite"],
)
def test_id_matching_rejects_incomplete_or_changed_population(change):
    inputs, predictions = populations()
    rows = deepcopy(predictions)
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[1] = rows[0]
    elif change == "extra":
        rows[0]["id"] = "different"
    else:
        key, value = {
            "label": ("label", 1),
            "source": ("source", "other"),
            "prompt": ("source_prompt_sha256", "changed"),
            "nonfinite": ("score", float("nan")),
        }[change]
        rows[0][key] = value
    with pytest.raises(ValueError):
        validate_prediction_membership(rows, inputs)


def test_compile_ahead_preserves_future_first_use_and_deduplicates_shapes():
    records = [
        {"update": 1, "tokens": 100},
        {"update": 2, "tokens": 200},
        {"update": 3, "tokens": 100},
        {"update": 4, "tokens": 200},
    ]
    assert future_shapes(records, 2) == [200, 100]
    assert future_shapes(records, 5) == []


def test_compile_ahead_rejects_empty_native_shape():
    with pytest.raises(ValueError):
        future_shapes([{"update": 2, "tokens": 0}], 2)
