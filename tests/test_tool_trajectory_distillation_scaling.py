import json
from collections import Counter

import pytest

from experiments.tool_trajectory_monitoring.distillation_scaling import (
    EFFECTIVE_BATCH_SIZE,
    MAX_LENGTH,
    PAPER_SCHEDULE,
    equal_capped_allocation,
    nested_count_selections,
    validate_jobs,
)
from experiments.tool_trajectory_monitoring.prepare_distillation_scaling import (
    make_jobs,
)
from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)


def synthetic_rows() -> list[dict[str, object]]:
    capacities = {
        "stride": 16,
        "gloom": 8,
        "cot_red_handed": 22,
        "bash_arena": 26,
        "bash_bench": 18,
    }
    rows = []
    source_row_index = 0
    for source, per_label in capacities.items():
        for label in (0, 1):
            for index in range(per_label):
                rows.append(
                    {
                        "dataset": f"tool_trajectory/{source}",
                        "source_dataset": source,
                        "index": f"{source}-{label}-{index}",
                        "label": label,
                        "source_row_index": source_row_index,
                    }
                )
                source_row_index += 1
    return rows


def test_equal_capped_allocation_is_exact_and_balanced() -> None:
    capacities = {
        (f"source-{index}", label): 100
        for index in range(5)
        for label in (0, 1)
    }
    allocation = equal_capped_allocation(capacities, 204)
    assert sum(allocation.values()) == 204
    assert set(allocation.values()) == {20, 21}

    capacities[("source-0", 0)] = 10
    capped = equal_capped_allocation(capacities, 204)
    assert capped[("source-0", 0)] == 10
    assert max(capped.values()) - min(
        value for key, value in capped.items() if key != ("source-0", 0)
    ) <= 1


def test_count_selections_are_exact_nested_and_seed_stable() -> None:
    rows = synthetic_rows()
    counts = (20, 50, 100, len(rows))
    first = nested_count_selections(rows, counts, seed=0)
    second = nested_count_selections(list(reversed(rows)), counts, seed=0)
    previous: set[str] = set()
    for count in counts:
        identities = {str(row["index"]) for row in first[count]}
        assert len(identities) == count
        assert previous <= identities
        assert identities == {str(row["index"]) for row in second[count]}
        previous = identities


def test_duplicate_selection_identities_fail_closed() -> None:
    rows = synthetic_rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate identities"):
        nested_count_selections(rows, [20], seed=0)


def test_frozen_jobs_are_seven_soft_only_rank128_conditions(tmp_path) -> None:
    jobs = make_jobs(tmp_path / "data", tmp_path / "results")
    validate_jobs(jobs)
    assert len(jobs) == 7
    assert Counter(job["target"] for job in jobs) == {"kimi_soft": 7}
    assert {job["max_steps"] for job in jobs if job["train_rows"] != 21_837} == set(
        PAPER_SCHEDULE.values()
    )
    mixed = next(job for job in jobs if job["train_rows"] == 21_837)
    assert mixed["num_train_epochs"] == 3.0
    assert mixed["selection_manifest"] is None


def test_training_command_cannot_enable_hard_or_completion_loss(tmp_path) -> None:
    job = make_jobs(tmp_path / "data", tmp_path / "results")[0]
    job.update(
        output_dir="/results/run",
        causal_adapter_dir="/results/run/causal_adapter",
        model_dir="/results/run/model",
    )
    command = training_command(job)
    assert "student.training.soft_loss_weight=1.0" in command
    assert "student.training.direct_loss_weight=0.0" in command
    assert "student.training.completion_loss_weight=0.0" in command
    assert f"student.max_length={MAX_LENGTH}" in command
    assert "student.lora.r=128" in command
    assert (
        f"student.training.gradient_accumulation_steps={EFFECTIVE_BATCH_SIZE}"
        in command
    )
    assert json.loads(json.dumps(job))["target"] == "kimi_soft"
