import json
import subprocess
from pathlib import Path

import pytest

from experiments.monitoring_objective_ablation import prepare, run_lambda
from experiments.monitoring_objective_ablation.core import (
    ARM_SPECS,
    CONTROL,
    select_winner,
    validate_loss_metadata,
)
from gleipnir.campaign_status import CampaignStatus

ROOT = Path(__file__).resolve().parents[1]


def test_loss_metadata_accepts_versioned_normalization_without_relaxing_weights():
    job = ARM_SPECS[0]
    legacy = {
        "completion_weight": 0.05,
        "completion_logits_mode": "selected_positions",
        "completion_projection_chunk_size": 128,
        "sequential_objective_backward": True,
        "direct_weight": 0.0,
        "pairwise_weight": 0.0,
        "soft_weight": 1.0,
        "soft_type": "bce",
        "mil_weight": 0.0,
        "mil_pooling": "logmeanexp",
        "mil_temperature": 1.0,
        "mil_top_k": 3,
        "mil_max_instances": 8,
    }
    corrected = {
        **legacy,
        "accumulation_policy": "explicit_microbatch_mean_v1",
        "model_accepts_loss_kwargs": False,
    }
    validate_loss_metadata(legacy, job)
    validate_loss_metadata(corrected, job)
    for override in (
        {"model_accepts_loss_kwargs": True},
        {"accumulation_policy": "unknown"},
        {"completion_weight": 0.20},
        {"unexpected_loss": 1.0},
    ):
        with pytest.raises(ValueError):
            validate_loss_metadata({**corrected, **override}, job)
    with pytest.raises(ValueError):
        validate_loss_metadata({**legacy, "model_accepts_loss_kwargs": False}, job)


def test_training_subset_is_ordered_and_fail_closed():
    jobs = [{"job_name": name} for name in ("a", "b", "c")]
    assert run_lambda.select_training_jobs(jobs, ["c", "a"]) == [jobs[0], jobs[2]]
    for names in ([], ["a", "a"], ["missing"]):
        with pytest.raises(ValueError):
            run_lambda.select_training_jobs(jobs, names)


def test_selected_training_preflights_accumulation_and_respects_gpu(
    tmp_path, monkeypatch
):
    job = {"job_name": "rationale", "kind": "rationale"}
    selection = tmp_path / "selection.jsonl"
    selection.write_text("{}\n{}\n")
    preflights, lanes = [], []

    def preflight(path, name, environment, **kwargs):
        row = prepare.read_jsonl(path)[0]
        preflights.append((row, environment, kwargs))
        adapter = Path(row["causal_adapter_dir"])
        adapter.mkdir(parents=True)
        (adapter / "training_metadata.json").write_text(
            json.dumps(
                {
                    "losses": {
                        "accumulation_policy": "explicit_microbatch_mean_v1",
                        "model_accepts_loss_kwargs": False,
                    }
                }
            )
        )

    monkeypatch.setattr(run_lambda, "run_training_job", preflight)
    monkeypatch.setattr(run_lambda, "train_lane", lambda *args: lanes.append(args))
    status = CampaignStatus(tmp_path / "status.json", [job], None)
    run_lambda.train_selected_jobs(
        [job], tmp_path / "jobs.jsonl", tmp_path, selection, [1], status, {}
    )
    row, environment, kwargs = preflights[0]
    assert row["train_rows"] == row["gradient_accumulation_steps"] == 2
    assert row["max_steps"] == 1
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert kwargs == {"preflight": True}
    assert lanes[0][:2] == ([job], 1)


def test_bad_accumulation_cannot_win_even_with_high_id_score():
    row = {
        **CONTROL,
        "job_name": "bad-scaling",
        "macro_pauroc_at_20": 0.99,
        "macro_auroc": 0.99,
        "objective_accumulation_valid": False,
    }
    assert select_winner([row])["selected_job"] is None


def test_rerun_preparation_preserves_recipe_and_original_artifacts(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(prepare, "validate_jobs", lambda jobs: None)
    jobs = [
        {
            "job_name": name,
            "kind": kind,
            "learning_rate": 2e-5,
            "model_dir": f"original/{name}/model",
            "causal_adapter_dir": f"original/{name}/causal",
            "output_dir": f"original/{name}",
        }
        for name, kind in [("rationale", "rationale"), ("mil", "mil")]
    ]
    source = tmp_path / "original" / "jobs.jsonl"
    prepare.atomic_write_jsonl(source, jobs)
    repaired = prepare.prepare_reruns(source, tmp_path / "repair", "rationale")
    assert repaired[1] == jobs[1]
    assert repaired[0]["learning_rate"] == jobs[0]["learning_rate"]
    assert repaired[0]["model_dir"].endswith("repair/runs/rationale/model")
    assert prepare.read_jsonl(source) == jobs
    with pytest.raises(ValueError, match="separate"):
        prepare.prepare_reruns(source, source.parent, "rationale")


def test_objective_evaluation_preserves_complete_id_provenance():
    objective = json.loads(
        (
            ROOT / "experiments/monitoring_objective_ablation/id_benchmark.json"
        ).read_text()
    )
    control = json.loads(
        (ROOT / "experiments/monitoring_lr_sweep/id_benchmark.json").read_text()
    )
    for key in (
        "input",
        "input_sha256",
        "source_input_sha256",
        "manifest",
        "manifest_sha256",
        "rows",
        "sources",
    ):
        assert objective["scope"][key] == control["scope"][key]
    assert objective["prompt"] == control["prompt"]
    assert objective["engine"] == control["engine"]


def test_lane_metadata_failure_is_published_and_stops_next_job(tmp_path, monkeypatch):
    jobs = [
        {"job_name": name, "causal_adapter_dir": str(tmp_path / name)}
        for name in ("a", "b")
    ]
    status_path = tmp_path / "status.json"
    status = CampaignStatus(status_path, jobs, None)
    called = []
    monkeypatch.setattr(
        run_lambda, "run_training_job", lambda *args, **kwargs: called.append(args[1])
    )

    def invalid_metadata(*args):
        raise ValueError("metadata drift")

    monkeypatch.setattr(run_lambda, "validate_training_metadata", invalid_metadata)
    with pytest.raises(ValueError, match="metadata drift"):
        run_lambda.train_lane(jobs, 1, tmp_path / "jobs.jsonl", status, {})
    recorded = json.loads(status_path.read_text())
    assert called == ["a"]
    assert recorded["failed_jobs"] == ["a"]
    assert recorded["completed_jobs"] == []
    assert recorded["active_jobs"] == []


def test_serving_parity_uses_bounded_canary_and_propagates_failure(
    tmp_path, monkeypatch
):
    calls = []
    fast, serving = {"KERNELS": "pinned"}, {"CUDA_VISIBLE_DEVICES": "1"}

    def run(command, environment):
        calls.append((command, environment))
        if (
            "experiments.tool_trajectory_monitoring.compare_distilled_ood_parity"
            in command
        ):
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(run_lambda, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        run_lambda.run_serving_parity(
            ROOT / "experiments/monitoring_objective_ablation/id_benchmark.json",
            tmp_path,
            fast,
            serving,
        )
    assert len(calls) == 3
    assert calls[0][1] is fast
    assert calls[1][1] is serving
    assert "--canary-only" in calls[1][0]
    assert "--include-base" in calls[1][0]
    assert "soft-rationale-w005" in calls[1][0]
