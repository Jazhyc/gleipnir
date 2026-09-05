import json
import subprocess
from pathlib import Path

import pytest

from experiments.monitoring_objective_ablation import prepare, run_lambda
from experiments.monitoring_objective_ablation.core import CONTROL, select_winner
from gleipnir.campaign_status import CampaignStatus

ROOT = Path(__file__).resolve().parents[1]


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
