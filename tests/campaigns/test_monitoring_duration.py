import json

import pytest
from omegaconf import OmegaConf

from experiments.monitoring_duration import run
from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)


def config():
    return OmegaConf.to_container(
        OmegaConf.load("experiments/monitoring_duration/config.yaml"), resolve=True
    )


def test_duration_jobs_only_change_declared_fields():
    cfg = config()
    jobs = run.make_jobs(cfg)
    originals = run.make_lr_jobs(run.Path(cfg["data_dir"]), run.Path(cfg["result_dir"]))
    assert len(jobs) == 2
    allowed = {
        "job_name",
        "design_role",
        "num_train_epochs",
        "save_steps",
        "completion_loss_weight",
        "mil_loss_weight",
        "output_dir",
        "causal_adapter_dir",
        "model_dir",
    }
    for job, original in zip(jobs, originals[:2], strict=True):
        assert {key for key in job if job[key] != original.get(key)} <= allowed
        assert job["num_train_epochs"] == 2
        assert job["save_steps"] == 544
        assert job["max_steps"] == -1
        command = training_command(job)
        assert "student.training.num_train_epochs=2.0" in command
        assert "student.training.completion_loss_weight=0.0" in command
        assert "student.training.mil_loss_weight=0.0" in command
        assert not any("init_adapter" in item or "resume" in item for item in command)


@pytest.mark.parametrize(
    "override",
    [
        {"epochs": 3},
        {"learning_rates": [2e-5]},
        {"seed": 1},
        {"strict_ood_consulted": True},
    ],
)
def test_duration_design_rejects_scope_changes(override):
    with pytest.raises(ValueError):
        run.make_jobs({**config(), **override})


def test_metadata_validation_checks_actual_metrics_schema(monkeypatch):
    metadata = {
        "losses": {"soft_weight": 1.0, "soft_type": "bce"},
        "train_metrics": {"train_loss": 0.2},
    }
    calls = []

    def validate(*args, **kwargs):
        calls.append(kwargs)
        return metadata

    monkeypatch.setattr(run, "validate_training_metadata", validate)
    job = run.make_jobs(config())[0]
    run.validate_completed(job)
    assert calls[-1]["expected_steps"] == 544
    run.validate_completed(job, preflight=True)
    assert calls[-1] == {"expected_steps": 1, "require_canary": True}
    metadata["train_metrics"]["train_loss"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        run.validate_completed(job)


def test_frozen_file_drift_stops_before_training(tmp_path, monkeypatch):
    file = tmp_path / "jobs.jsonl"
    file.write_text("changed")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {str(file): "wrong"}}))
    monkeypatch.setattr(
        run, "run_training_job", lambda *a, **kw: pytest.fail("must not train")
    )
    with pytest.raises(ValueError, match="contract drift"):
        run.execute(tmp_path, "test")
