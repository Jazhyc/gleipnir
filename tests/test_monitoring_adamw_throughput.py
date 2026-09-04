import json
from pathlib import Path

from experiments.monitoring_adamw_throughput.core import (
    CONDITIONS,
    select_recipe,
    validate_jobs,
)
from experiments.monitoring_adamw_throughput.prepare import make_jobs
from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)


def test_jobs_hold_everything_except_adamw_implementation_fixed(
    tmp_path: Path,
) -> None:
    selection = tmp_path / "selection.jsonl"
    selection.write_text("{}\n")
    jobs = make_jobs(tmp_path / "data", tmp_path / "results", selection)
    validate_jobs(jobs)
    assert [job["trainer_optim"] for job in jobs] == [
        condition["trainer_optim"] for condition in CONDITIONS
    ]
    assert {job["micro_batch_size"] for job in jobs} == {1}
    assert {job["gradient_accumulation_steps"] for job in jobs} == {32}
    assert "student.training.optim=adamw_torch_fused" in training_command(jobs[1])


def test_fused_optimizer_requires_three_percent_gain() -> None:
    rows = [
        {"job_name": "adamw-torch", "train_samples_per_second": 1.0},
        {"job_name": "adamw-torch-fused", "train_samples_per_second": 1.029},
    ]
    assert select_recipe(rows)["selected_job"] == "adamw-torch"
    rows[1]["train_samples_per_second"] = 1.03
    assert select_recipe(rows)["selected_job"] == "adamw-torch-fused"


def test_config_matches_conditions() -> None:
    config = json.loads(
        Path("experiments/monitoring_adamw_throughput/config.json").read_text()
    )
    assert [condition["name"] for condition in config["conditions"]] == [
        condition["job_name"] for condition in CONDITIONS
    ]
