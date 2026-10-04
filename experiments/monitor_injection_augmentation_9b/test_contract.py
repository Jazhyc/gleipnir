"""Focused replication, baseline-integrity and checkpointed reuse checks."""

import copy
import functools
import json
from types import SimpleNamespace

import pytest

from experiments.monitor_injection_augmentation_9b.prepare import (
    cached_scores,
    configuration,
)
from experiments.monitor_injection_augmentation_9b.train import make_job
from experiments.monitoring_hard_labels.test_validated_startup import reference_metadata
from gleipnir import validated_startup
from gleipnir.monitoring_campaign_data import digest
from gleipnir.monitoring_training_command import training_command


def checkpointed_reference():
    m = reference_metadata()
    m.update(
        model="Qwen/Qwen3.5-9B",
        gradient_checkpointing=True,
        gradient_checkpointing_policy="all",
        checkpointed_layer_indices=list(range(32)),
        selective_torch_compile={
            "policy": "checkpointed_full_attention_and_linear_shell"
        },
    )
    return m


def test_checkpointed_9b_reuse_preserves_failed_strict_receipt(tmp_path):
    p = tmp_path / "metadata.json"
    m = checkpointed_reference()
    p.write_text(json.dumps(m))
    ref = validated_startup.validation_reference(p)
    assert ref["gradient_checkpointing"] and not ref["performed_this_run"]
    assert not ref["reference_backend"]["passed"]
    m["checkpointed_layer_indices"] = [0]
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="checkpointed 9B"):
        validated_startup.validation_reference(p)


def test_checkpoint_reuse_rejects_live_checkpoint_mismatch(monkeypatch):
    ref = {"gradient_checkpointing": True}
    with pytest.raises(ValueError, match="checkpointing differs"):
        validated_startup.install_validated_flashqla(
            SimpleNamespace(is_gradient_checkpointing=False), ref
        )
    modules = [
        SimpleNamespace(
            gradient_checkpointing=True,
            _gradient_checkpointing_func=functools.partial(
                lambda: None, use_reentrant=True
            ),
        )
    ]
    model = SimpleNamespace(is_gradient_checkpointing=True, modules=lambda: modules)
    with pytest.raises(ValueError, match="nonreentrant"):
        validated_startup.install_validated_flashqla(model, ref)


def test_job_retains_validated_9b_recipe_and_regular_lr():
    from experiments.monitor_injection_augmentation_9b.prepare import ROOT

    config = configuration()
    prior = json.loads((ROOT / config["prior_job"]).read_text())
    job = make_job(config, prior)
    changed = {k for k in prior if prior[k] != job[k]}
    assert changed <= {
        "job_name",
        "student_rows",
        "output_dir",
        "model_dir",
        "causal_adapter_dir",
        "hydra_log_dir",
        "soft_targets",
    }
    assert job["learning_rate"] == 5e-5 and job["effective_batch_size"] == 32
    command = training_command(job)
    assert "++student.training.nonreentrant_checkpointing=true" in command
    assert "student.training.gradient_checkpointing=true" in command
    bad = copy.deepcopy(prior)
    bad["learning_rate"] = 2e-5
    with pytest.raises(ValueError, match="recipe drift"):
        make_job(config, bad)


def test_baseline_rejects_changed_prompts_labels_and_logits(tmp_path):
    import math

    template = "{{ messages[0].content }}"
    row = {
        "id": "x",
        "prompt": "trajectory",
        "metadata": {"ground_truth": 0, "source_dataset": "source"},
    }
    cached = {
        "id": "x",
        "label": 0,
        "source": "source",
        "margin_prompt_sha256": digest("trajectoryPrediction:"),
        "logprob_0": -0.1,
        "logprob_1": -1.1,
        "score": 1 / (1 + math.exp(1)),
    }
    p = tmp_path / "scores.jsonl"
    p.write_text(json.dumps(cached) + "\n")
    assert len(cached_scores(p, [row], template)) == 1
    for key, value in (("label", 1), ("margin_prompt_sha256", "wrong"), ("score", 0.7)):
        wrong = {**cached, key: value}
        p.write_text(json.dumps(wrong) + "\n")
        with pytest.raises(ValueError, match="drift"):
            cached_scores(p, [row], template)
