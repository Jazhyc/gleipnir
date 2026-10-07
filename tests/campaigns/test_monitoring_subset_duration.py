import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from experiments.monitoring_lr_sweep.prepare import make_jobs as lr_jobs
from experiments.monitoring_subset_duration import run
from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)
from gleipnir.staged_lanes import run_staged_lanes


@pytest.fixture
def config(monkeypatch):
    cfg = OmegaConf.to_container(
        OmegaConf.load("experiments/monitoring_subset_duration/config.yaml")
    )
    cfg["subset_parents_with_prefix"] = 1270
    base = next(
        j for j in lr_jobs(Path("data"), Path("results")) if j["learning_rate"] == 2e-5
    )
    sources = {
        cfg["standard_jobs"]: [
            {
                **base,
                "job_name": cfg["standard_job"],
                "train_rows": 1738,
                "selection_manifest": "selection.jsonl",
                "selection_sha256": "fixed",
                "mil_loss_weight": 0.0,
            }
        ],
        cfg["mil_jobs"]: [
            {
                **base,
                "job_name": cfg["mil_job"],
                "mil_loss_weight": 0.25,
                "mil_pooling": "logmeanexp",
                "mil_temperature": 1.0,
                "mil_max_instances": 8,
                "gradient_checkpointing_policy": "all",
                "selective_torch_compile_policy": "linear_attention_shells_only",
            }
        ],
        cfg["prefix_jobs"]: [
            {**base, "job_name": cfg["prefix_job"], "prefix_loss_weight": 0.1}
        ],
    }
    monkeypatch.setattr(run, "read_jsonl", lambda p: sources[str(p)])
    return cfg


def test_nine_fresh_cells_preserve_selection_and_objectives(config):
    jobs = run.make_jobs(config)
    assert len(jobs) == len({j["job_name"] for j in jobs}) == 9
    for job in jobs:
        assert job["train_rows"] == 1738
        assert job["selection_manifest"] == "selection.jsonl"
        assert job["selection_sha256"] == "fixed"
        assert (
            job["expected_steps"] == job["save_steps"] == 55 * job["num_train_epochs"]
        )
        assert job["max_steps"] == -1
        assert job["completion_loss_weight"] == 0
        assert job["prefix_loss_weight"] == (0.1 if job["objective"] == "prefix" else 0)
        assert job["mil_loss_weight"] == (0.25 if job["objective"] == "mil" else 0)
        assert not any(
            "resume" in item or "init_adapter" in item for item in training_command(job)
        )


def test_lane_plan_has_explicit_balanced_second_stage(config):
    stages = run.lane_plan(run.make_jobs(config))
    assert [[j["num_train_epochs"] for j in lane] for _, lane in stages[0]] == [
        [2, 3, 5],
        [2, 3, 5],
    ]
    assert [[j["num_train_epochs"] for j in lane] for _, lane in stages[1]] == [
        [5],
        [2, 3],
    ]


@pytest.mark.parametrize(
    "override",
    [
        {"epochs": [5]},
        {"seed": 1},
        {"train_rows": 8688},
        {"prefix_stage_after": "training_only"},
    ],
)
def test_scope_drift_fails(config, override):
    with pytest.raises(ValueError):
        run.make_jobs({**config, **override})


def test_stages_wait_for_all_lanes_and_completion_audit():
    events = []

    def worker(gpu, value):
        if value == "prefix":
            assert "audit0" in events
            assert "soft" in events and "mil" in events
        events.append(value)

    run_staged_lanes(
        [["soft", "mil"], ["prefix"]],
        worker,
        lambda stage: events.append(f"audit{stage}"),
    )
    assert events[-1] == "audit1"


@pytest.mark.parametrize("fail_audit", [False, True])
def test_stage_failure_prevents_prefix_launch(fail_audit):
    events = []

    def worker(gpu, value):
        events.append(value)
        if not fail_audit:
            raise RuntimeError("lane failed")

    def audit(stage):
        raise RuntimeError("evaluation incomplete")

    with pytest.raises(RuntimeError):
        run_staged_lanes([["soft"], ["prefix"]], worker, audit)
    assert "prefix" not in events


def test_frozen_drift_fails_before_gpu_work(tmp_path):
    path = tmp_path / "jobs.jsonl"
    path.write_text("changed")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {str(path): "wrong"}}))
    with pytest.raises(ValueError, match="frozen contract drift"):
        run.execute(tmp_path, "test")
