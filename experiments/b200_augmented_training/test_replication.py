"""Frozen recipe, new-adapter gating and full-epoch coverage checks."""

import copy

import pytest

from experiments.b200_augmented_training import campaign
from experiments.b200_augmented_training.campaign import (
    configuration,
    make_job,
    profile,
)
from experiments.b200_augmented_training.evaluate import agreement, attach_inputs
from gleipnir.campaigns.training_command import training_command


def test_augmented_recipe_is_fresh_full_epoch_native_fp4():
    config = configuration()
    job = make_job(config, profile(config)["recipe"])
    command = training_command(job)
    assert (
        job["native_fp4_mlp"] and job["packed_attention_backend"] == "flash_attention_4"
    )
    assert (
        job["gated_delta_backend"] == "flashqla" and not job["gradient_checkpointing"]
    )
    assert job["soft_loss_weight"] == 1 and job["direct_loss_weight"] == 0
    assert job["max_steps"] == -1 and job["num_train_epochs"] == 1
    assert (
        job["learning_rate"] == 5e-5 and job["rank"] == 128 and job["lora_alpha"] == 256
    )
    assert "student.training.max_steps=-1" in command
    assert "++student.training.native_fp4_mlp=true" in command
    assert job["startup_validation_reference_sha256"] == campaign.REFERENCE_SHA256
    assert "monitor_injection_augmentation/student_rows.jsonl" in job["student_rows"]
    assert "student_injection_awareness/soft_targets.jsonl" in job["soft_targets"]


def test_initialization_file_checksum_and_loaded_tensor_fingerprint_are_distinct():
    config = configuration()
    job = make_job(config, profile(config)["recipe"])
    file_checksum = config["inputs"]["initial_weights"]["sha256"]
    tensor_fingerprint = config["model"]["initial_master_sha256"]
    assert file_checksum != tensor_fingerprint
    assert len(file_checksum) == len(tensor_fingerprint) == 64
    assert job["expected_initial_master_sha256"] == tensor_fingerprint
    assert (
        f"++student.training.expected_initial_master_sha256={tensor_fingerprint}"
        in training_command(job)
    )


def test_new_adapter_gate_rejects_score_drift_and_zero_adapter_effect():
    limits = configuration()["parity"]
    scores = [0.1, 0.2, 0.8, 0.9]
    base = [0.4, 0.4, 0.4, 0.4]
    assert agreement([0.101, 0.202, 0.803, 0.904], scores, base, limits)["passed"]
    assert not agreement([0.3, 0.4, 0.6, 0.7], scores, base, limits)["passed"]
    assert not agreement(scores, scores, scores, limits)["passed"]
    with pytest.raises(ValueError, match="coverage"):
        agreement(scores[:-1], scores, base, limits)


@pytest.mark.parametrize(
    "scores", [[0.1, float("nan"), 0.8, 0.9], [0.1, 1.2, 0.8, 0.9], [0.5] * 4]
)
def test_invalid_or_constant_scores_cannot_pass_new_adapter_gate(scores):
    receipt = agreement(
        scores, [0.1, 0.2, 0.8, 0.9], [0.4] * 4, configuration()["parity"]
    )
    assert not receipt["passed"]


def test_saved_labels_cannot_be_overwritten_during_input_attachment():
    rows = [
        {"id": "example", "metadata": {"ground_truth": 0, "source_dataset": "source"}}
    ]
    assert (
        attach_inputs([{"id": "example", "score": 0.2}], rows)[0]["ground_truth"] == 0
    )
    with pytest.raises(ValueError, match="metadata"):
        attach_inputs([{"id": "example", "score": 0.2, "ground_truth": 1}], rows)
    with pytest.raises(ValueError, match="identity"):
        attach_inputs([{"id": "other", "score": 0.2}], rows)


def test_duplicate_physical_index_fails_despite_unchanged_update_and_token_totals(
    monkeypatch,
):
    monkeypatch.setattr(
        campaign, "validate_training_metadata", lambda *args, **kwargs: None
    )
    config = {**configuration(), "expected_training_tokens": 272}
    sizes = [32] * 271 + [16]
    metadata = {
        "adaptive_microbatching": {
            "logical_batch_sizes": sizes,
            "records": [
                {
                    "update": i,
                    "logical_indices": list(range(size)),
                    "tokens": 1,
                    "padded_tokens": 1,
                }
                for i, size in enumerate(sizes, 1)
            ],
        },
        "train_metrics": {"epoch": 1.0},
        "optimizer_step_timing": {"durations_seconds": [1.0] * 272},
    }
    campaign.validate_completion(metadata, config, {})
    broken = copy.deepcopy(metadata)
    broken["adaptive_microbatching"]["records"][17]["logical_indices"][-1] = 0
    with pytest.raises(ValueError, match="missing or repeated"):
        campaign.validate_completion(broken, config, {})
