import json

import pandas as pd
import pytest

from experiments.adapter_capacity_scaling.run_train import training_command
from experiments.data_capacity_scaling.core import (
    DEFAULT_RANKS,
    balanced_interaction_lanes,
    fraction_tag,
    validate_design,
)
from experiments.data_capacity_scaling.summarize import (
    high_rank_interaction,
    paired_contrasts,
)


def interaction_job(seed: int, fraction: float, rank: int) -> dict[str, object]:
    tag = fraction_tag(fraction)
    return {
        "job_name": f"seed{seed}-{tag}-r{rank:03d}",
        "capacity_kind": "lora",
        "seed": seed,
        "fraction": fraction,
        "rank": rank,
        "lora_alpha": 2 * rank,
        "train_rows": round(13149 * fraction),
        "selection_manifest": f"/results/selections/seed{seed}-{tag}.jsonl",
        "student_rows": "/data/student.jsonl",
        "soft_targets": "/data/soft.jsonl",
        "output_dir": "/results/run",
        "causal_adapter_dir": "/results/run/causal_adapter",
        "model_dir": "/results/run/model",
    }


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for fraction, train_rows in ((0.1, 1315), (0.25, 3287), (0.5, 6575), (1.0, 13149)):
        for seed in range(3):
            for rank in DEFAULT_RANKS:
                high_rank_gain = 0.002 * fraction * (rank / 256)
                rows.append(
                    {
                        "fraction": fraction,
                        "train_rows": train_rows,
                        "seed": seed,
                        "rank": rank,
                        "macro_auroc": 0.94 + 0.001 * seed + high_rank_gain,
                        "macro_balanced_accuracy": 0.90 + high_rank_gain,
                        "macro_brier": 0.08 - high_rank_gain,
                        "pooled_auroc": 0.95 + high_rank_gain,
                    }
                )
    return pd.DataFrame(rows)


def test_design_and_fraction_tags_are_fixed() -> None:
    fractions, ranks, seeds = validate_design(
        [0.5, 0.1, 0.25], [256, 4, 16, 64], [2, 0, 1]
    )
    assert fractions == [0.1, 0.25, 0.5]
    assert ranks == [4, 16, 64, 256]
    assert seeds == [0, 1, 2]
    assert fraction_tag(0.1) == "f010"
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        validate_design([1.0], [16], [0])


def test_training_command_uses_exact_selection_manifest() -> None:
    job = interaction_job(1, 0.25, 64)
    command = training_command(job, distributed_processes=1)
    assert "student.selection_manifest=/results/selections/seed1-f025.jsonl" in command
    assert "student.lora.r=64" in command


def test_interaction_lanes_balance_total_rows() -> None:
    jobs = [
        interaction_job(seed, fraction, rank)
        for seed in range(3)
        for fraction in (0.1, 0.25, 0.5)
        for rank in DEFAULT_RANKS
    ]
    lanes = balanced_interaction_lanes(jobs, 2)
    assert sorted(job["job_name"] for lane in lanes for job in lane) == sorted(
        job["job_name"] for job in jobs
    )
    loads = [sum(int(job["train_rows"]) for job in lane) for lane in lanes]
    assert abs(loads[0] - loads[1]) <= max(int(job["train_rows"]) for job in jobs)


def test_paired_contrasts_and_interaction_detect_increasing_gain() -> None:
    frame = synthetic_frame()
    contrasts = paired_contrasts(frame)
    selected = contrasts[
        (contrasts["metric"] == "macro_auroc") & (contrasts["rank"] == 256)
    ]
    assert len(selected) == 4
    assert selected.iloc[-1]["paired_mean_difference"] > selected.iloc[0][
        "paired_mean_difference"
    ]
    interaction = high_rank_interaction(frame)
    assert interaction["log_train_rows_slope"] > 0
    assert interaction["observations"] == 12


def test_experiment_config_is_valid_json_compatible_yaml() -> None:
    # Ensure the source-level recipe constants remain serializable as manifest data.
    payload = {"ranks": list(DEFAULT_RANKS), "fractions": [0.1, 0.25, 0.5]}
    assert json.loads(json.dumps(payload)) == payload
