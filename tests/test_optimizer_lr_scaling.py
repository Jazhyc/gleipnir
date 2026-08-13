import pytest
import pandas as pd

from experiments.adapter_capacity_scaling.run_train import training_command
from experiments.optimizer_lr_scaling.core import (
    learning_rate_tag,
    paired_optimizer_lanes,
    validate_learning_rates,
)
from experiments.optimizer_lr_scaling.summarize import (
    CONTRAST_METRICS,
    paired_contrasts,
)


def optimizer_job(optimizer: str, learning_rate: float) -> dict[str, object]:
    return {
        "job_name": f"{optimizer}-lr{learning_rate_tag(learning_rate)}",
        "capacity_kind": "lora",
        "seed": 0,
        "rank": 64,
        "lora_alpha": 128,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.03,
        "num_train_epochs": 2.0,
        "weight_decay": 0.0,
        "muon_adjust_lr_fn": "match_rms_adamw",
        "train_rows": 3287,
        "selection_manifest": "/data/train.jsonl",
        "student_rows": "/data/student.jsonl",
        "soft_targets": "/data/soft.jsonl",
        "output_dir": "/results/run",
        "causal_adapter_dir": "/results/run/causal_adapter",
        "model_dir": "/results/run/model",
    }


def test_learning_rate_grid_and_tags_are_stable() -> None:
    assert validate_learning_rates([2e-4, 1e-5, 5e-5]) == [1e-5, 5e-5, 2e-4]
    assert learning_rate_tag(1e-5) == "1em05"
    assert learning_rate_tag(2e-4) == "2em04"
    with pytest.raises(ValueError, match="positive"):
        learning_rate_tag(0.0)


def test_optimizer_lanes_pair_identical_rates() -> None:
    rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
    jobs = [
        optimizer_job(optimizer, rate)
        for rate in rates
        for optimizer in ("adamw", "muon")
    ]

    adamw, muon = paired_optimizer_lanes(jobs)

    assert [job["learning_rate"] for job in adamw] == [
        job["learning_rate"] for job in muon
    ]
    assert adamw[0]["learning_rate"] == 5e-5
    assert all(job["optimizer"] == "adamw" for job in adamw)
    assert all(job["optimizer"] == "muon" for job in muon)


def test_muon_job_uses_shared_scheduled_base_rate() -> None:
    command = training_command(optimizer_job("muon", 5e-5), distributed_processes=1)

    assert "student.training.optimizer=muon" in command
    assert "student.training.learning_rate=5e-05" in command
    assert "student.training.lr_scheduler_type=linear" in command
    assert "student.training.muon_adjust_lr_fn=match_rms_adamw" in command
    assert not any("muon_learning_rate" in value for value in command)


def test_paired_contrasts_accept_multiple_metric_columns() -> None:
    rows = []
    for optimizer, value in (("adamw", 1.0), ("muon", 1.25)):
        row = {
            "learning_rate": 5e-5,
            "optimizer": optimizer,
            **{metric: value for metric in CONTRAST_METRICS},
        }
        rows.append(row)

    contrasts = paired_contrasts(pd.DataFrame(rows))

    assert len(contrasts) == 1
    assert contrasts.loc[0, "learning_rate"] == pytest.approx(5e-5)
    for metric in CONTRAST_METRICS:
        assert contrasts.loc[0, f"muon_minus_adamw_{metric}"] == pytest.approx(
            0.25
        )
