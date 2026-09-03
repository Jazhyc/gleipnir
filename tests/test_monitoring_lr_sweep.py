import json
from pathlib import Path

from experiments.monitoring_lr_sweep.core import (
    LEARNING_RATES,
    expected_job_names,
    learning_rate_lanes,
    select_screen_winner,
    validate_jobs,
)
from experiments.monitoring_lr_sweep.prepare import make_jobs
from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (
    validate_config,
)
from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)


def test_jobs_freeze_monitoring_only_one_seed_lr_grid(tmp_path: Path) -> None:
    jobs = make_jobs(tmp_path / "data", tmp_path / "results")
    validate_jobs(jobs)
    assert [job["learning_rate"] for job in jobs] == list(LEARNING_RATES)
    assert [job["job_name"] for job in jobs] == expected_job_names()
    assert {job["seed"] for job in jobs} == {0}
    assert {job["data_scope"] for job in jobs} == {"monitoring_only"}
    assert {job["deception_rows"] for job in jobs} == {0}
    assert {job["micro_batch_size"] for job in jobs} == {2}
    assert {job["gradient_accumulation_steps"] for job in jobs} == {16}
    assert {job["require_causal_conv1d"] for job in jobs} == {True}
    assert {job["fla_disable_backend_dispatch"] for job in jobs} == {True}
    assert "student.training.require_causal_conv1d=true" in training_command(jobs[0])


def test_two_lanes_prioritize_control_and_nearby_rate(tmp_path: Path) -> None:
    jobs = make_jobs(tmp_path / "data", tmp_path / "results")
    lanes = learning_rate_lanes(jobs)
    assert [[job["learning_rate"] for job in lane] for lane in lanes] == [
        [5e-5, 2e-5, 1e-5],
        [1e-4, 2e-4],
    ]
    assert {job["job_name"] for lane in lanes for job in lane} == set(
        expected_job_names()
    )


def metric_row(rate: float, pauroc: float, brier: float = 0.10) -> dict:
    return {
        "job_name": f"job-{rate}",
        "learning_rate": rate,
        "macro_pauroc_at_20": pauroc,
        "macro_auroc": pauroc + 0.05,
        "macro_brier": brier,
        "source_pauroc_at_20": {"test_stride": pauroc, "gloom_exfiltration": pauroc},
    }


def test_selection_requires_practical_gain_and_guardrails() -> None:
    rows = [metric_row(rate, 0.80) for rate in LEARNING_RATES]
    rows[0] = metric_row(1e-5, 0.804)
    rows[1] = metric_row(2e-5, 0.807)
    rows[2] = metric_row(5e-5, 0.800)
    rows[3] = metric_row(1e-4, 0.810, brier=0.106)
    rows[4] = metric_row(2e-4, 0.806)
    selected = select_screen_winner(rows)
    assert selected["selected_learning_rate"] == 2e-5
    assert selected["control_retained"] is False


def test_id_benchmark_is_frozen_for_all_five_jobs() -> None:
    path = Path("experiments/monitoring_lr_sweep/id_benchmark.json")
    config = json.loads(path.read_text())
    validate_config(config)
    group = config["model_groups"]["4b"]
    assert group["expected_jobs"] == expected_job_names()
    assert group["evaluate_base"] is False
    assert config["engine"]["gdn_prefill_backend"] == "flashinfer"
