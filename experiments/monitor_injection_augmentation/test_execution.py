"""Verify matched training, held-out payload provenance and paired diagnostics."""

import pytest
import yaml

from experiments.monitor_injection_augmentation.metrics import (
    clean_threshold,
    synthetic_metrics,
)
from experiments.monitor_injection_augmentation.prepare import CONFIG, template_bank
from experiments.monitor_injection_augmentation.prepare_eval import synthetic_views
from experiments.monitor_injection_augmentation.train import make_job
from experiments.monitoring_hard_labels.prepare import resolved_profile
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from gleipnir.monitoring_campaign_data import digest, trajectory_from_prompt
from gleipnir.monitoring_training_command import training_command
from gleipnir.transcript_injection import remove_message


def test_matched_training_job_uses_only_original_soft_targets():
    job = make_job(yaml.safe_load(CONFIG.read_text()), resolved_profile()["recipe"])
    assert job["train_rows"] == 8688 and job["num_train_epochs"] == 1
    assert job["learning_rate"] == 5e-5
    assert job["soft_loss_weight"] == 1
    assert job["direct_loss_weight"] == job["completion_loss_weight"] == 0
    assert not job["gradient_checkpointing"]
    assert job["max_length"] == 29696
    command = training_command(job)
    assert "student.training.soft_loss_weight=1.0" in command
    assert "student.training.direct_loss_weight=0.0" in command
    assert not any("task_mixture" in c for c in command)


@pytest.mark.parametrize("headers", [("USER", "ASSISTANT"), ("user", "assistant")])
def test_synthetic_id_is_reversible_and_uses_only_heldout_families(headers):
    template = load_prompt_set().student
    trajectory = f"[{headers[0]}]\nDo the task.\n\n[{headers[1]}]\nTask completed.\n"
    prompt = template.render(trajectory)
    rows = [
        {
            "id": str(i),
            "prompt": prompt,
            "metadata": {
                "source_dataset": "source",
                "ground_truth": i % 2,
                "trajectory_sha256": digest(trajectory),
                "rendered_prompt_sha256": digest(prompt),
            },
        }
        for i in range(24)
    ]
    result = synthetic_views(rows, template_bank(), 1000)
    assert result == synthetic_views(rows, template_bank(), 1000)
    train_families = {
        t["family"] for t in template_bank()["templates"] if t["split"] == "train"
    }
    for old, new in zip(rows, result, strict=True):
        meta = new["metadata"]
        assert meta["ground_truth"] == old["metadata"]["ground_truth"]
        assert meta["base_id"] == old["id"]
        assert meta["augmentation"]["template_family"] not in train_families
        recovered = remove_message(
            trajectory_from_prompt(new["prompt"], meta["trajectory_sha256"]),
            meta["augmentation"],
        )
        assert recovered == trajectory


def test_clean_threshold_keeps_strict_tie_policy():
    threshold = clean_threshold([0.1] * 99 + [0.9])
    assert threshold == 0.1
    assert sum(s > threshold for s in [0.1] * 99 + [0.9]) == 1
    with pytest.raises(ValueError):
        clean_threshold([])


def test_paired_metrics_use_clean_threshold_and_detect_class_specific_flips():
    clean = [
        {"id": str(i), "label": i, "score": (0.1, 0.9)[i], "source_dataset": "source"}
        for i in (0, 1)
    ]
    injected = [
        {
            **r,
            "base_id": r["id"],
            "id": r["id"] + "/injected",
            "score": 1 - r["score"],
            "augmentation": {
                "role": "tool",
                "template_family": "heldout",
                "demanded_decision": 1,
                "realized_position": 0.5,
            },
        }
        for r in clean
    ]
    report = synthetic_metrics(clean, injected)
    pooled = report["views"]["pooled"]
    assert report["clean_source_thresholds"]["source"]["threshold"] == 0.1
    assert pooled["label_0"]["correct_to_wrong_rate"] == 1
    assert pooled["label_1"]["correct_to_wrong_rate"] == 1
    assert pooled["label_0"]["clean_calibrated_injected_alarm_rate"] == 1
    assert pooled["label_0"]["mean_delta_p_harmful"] == pytest.approx(0.8)
    assert pooled["label_1"]["mean_delta_p_harmful"] == pytest.approx(-0.8)
    with pytest.raises(ValueError, match="coverage"):
        synthetic_metrics(clean, injected[:1])
    with pytest.raises(ValueError, match="label/source"):
        synthetic_metrics(clean, [{**injected[0], "label": 1}, injected[1]])
