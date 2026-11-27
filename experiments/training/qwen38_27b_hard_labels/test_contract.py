"""The hard arm may differ from the soft 27B seed 0 only in declared fields."""

from pathlib import Path

import yaml

from experiments.training.qwen38_27b_hard_labels.prepare import (
    ARM,
    FIELDS,
    OOD_CONTEXT,
    PARENT,
    ROOT,
    arm_config,
    scoring_config,
)
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.campaigns.training_command import training_command

ALLOWED = {
    "campaign_id",
    "soft_loss_weight",
    "direct_loss_weight",
    "gpu_uuid",
    "merged_model",
    "extra_sources",
    *FIELDS,
}


def parent() -> dict:
    return yaml.safe_load((ROOT / "data" / PARENT / "config.yaml").read_text())


def test_only_objective_identity_and_serving_hash_change():
    base = parent()
    assert (base["soft_loss_weight"], base["direct_loss_weight"]) == (1.0, 0.0)
    config = arm_config(base, "GPU-test", "0" * 64)
    changed = {k for k in config if config[k] != base.get(k)} - {"inputs"}
    assert changed <= ALLOWED
    assert config["campaign_id"] == ARM and config["seed"] == 0
    assert (config["soft_loss_weight"], config["direct_loss_weight"]) == (0.0, 1.0)
    for name, spec in config["inputs"].items():
        if name != "serving_selection":
            assert spec == base["inputs"][name]
    for key in (
        "profile",
        "learning_rate",
        "model",
        "logical_batch_size",
        "base_files",
        "full_startup_validation",
    ):
        assert config[key] == base[key]


def test_ood_context_binds_the_hard_arm():
    config = scoring_config(arm_config(parent(), "GPU-test", "0" * 64))
    assert config["parent_campaign"] == f"results/{ARM}"
    assert config["score_splits"] == ["ood"] and config["evaluations"] == []
    assert Path(config["merged_model"]).name == OOD_CONTEXT
    assert config["soft_loss_weight"] == 0.0


def test_no_kimi_artifact_reaches_the_trainer():
    config = arm_config(parent(), "GPU-test", "0" * 64)
    job = Campaign(ROOT, ROOT / "data" / ARM / "config.yaml", config).job()
    command = training_command(job)
    assert "student.soft_teacher_artifact=null" in command
    assert "student.training.direct_loss_weight=1.0" in command
