import json
from pathlib import Path

from experiments.monitoring_training_throughput.core import (
    CONDITIONS,
    SELECTION_ROWS,
    select_recipe,
    stable_stratified_selection,
    validate_jobs,
)
from experiments.monitoring_training_throughput.prepare import make_jobs
from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)


def test_selection_is_exact_stable_and_stratified() -> None:
    records = [
        {
            "dataset": f"dataset-{index % 2}",
            "index": index,
            "label": index % 2,
        }
        for index in range(1_000)
    ]
    first = stable_stratified_selection(records, count=100, seed=7)
    second = stable_stratified_selection(list(reversed(records)), count=100, seed=7)
    assert len(first) == 100
    assert {(row["dataset"], row["index"]) for row in first} == {
        (row["dataset"], row["index"]) for row in second
    }
    assert {row["label"] for row in first} == {0, 1}


def test_jobs_hold_examples_and_effective_batch_fixed(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    selection.write_text("{}\n")
    jobs = make_jobs(tmp_path / "data", tmp_path / "results", selection)
    validate_jobs(jobs)
    assert len(jobs) == len(CONDITIONS)
    assert {job["train_rows"] for job in jobs} == {SELECTION_ROWS}
    assert {
        job["micro_batch_size"] * job["gradient_accumulation_steps"] for job in jobs
    } == {32}
    grouped = next(job for job in jobs if job["job_name"] == "mb2-length-grouped")
    assert grouped["train_sampling_strategy"] == "group_by_length"
    assert (
        "student.training.train_sampling_strategy=group_by_length"
        in training_command(grouped)
    )


def test_recipe_selection_requires_five_percent_gain() -> None:
    rows = []
    for condition, rate in zip(CONDITIONS, (0.49, 0.50, 0.524), strict=True):
        rows.append(
            {
                **condition,
                "train_samples_per_second": rate,
                "direct_padding_fraction": 0.0,
            }
        )
    assert select_recipe(rows)["selected_job"] == "mb2-random"
    rows[-1]["train_samples_per_second"] = 0.53
    assert select_recipe(rows)["selected_job"] == "mb2-length-grouped"


def test_config_matches_frozen_conditions() -> None:
    config = json.loads(
        Path("experiments/monitoring_training_throughput/config.json").read_text()
    )
    assert config["selection_rows"] == SELECTION_ROWS
    assert [condition["name"] for condition in config["conditions"]] == [
        condition["job_name"] for condition in CONDITIONS
    ]
