import pandas as pd
import pytest

from experiments.adapter_capacity_scaling.run_train import training_command
from experiments.optimizer_full_data_confirmation.core import (
    CONFIRMATION_LEARNING_RATES,
    CONFIRMATION_SEEDS,
    confirmation_lanes,
    validate_confirmation_jobs,
)
from experiments.optimizer_full_data_confirmation.summarize import (
    METRICS,
    paired_confirmation_contrasts,
    selection_summary,
)


def confirmation_job(seed: int, learning_rate: float) -> dict[str, object]:
    return {
        "job_name": f"seed{seed}-muon-{learning_rate}",
        "capacity_kind": "lora",
        "seed": seed,
        "fraction": 1.0,
        "rank": 64,
        "lora_alpha": 128,
        "optimizer": "muon",
        "learning_rate": learning_rate,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.03,
        "num_train_epochs": 2.0,
        "weight_decay": 0.0,
        "muon_adjust_lr_fn": "match_rms_adamw",
        "train_rows": 13149,
        "student_rows": "/data/student.jsonl",
        "soft_targets": "/data/soft.jsonl",
        "validation": "/data/validation.jsonl",
        "output_dir": "/results/run",
        "causal_adapter_dir": "/results/run/causal_adapter",
        "model_dir": "/results/run/model",
    }


def confirmation_jobs() -> list[dict[str, object]]:
    return [
        confirmation_job(seed, learning_rate)
        for seed in CONFIRMATION_SEEDS
        for learning_rate in CONFIRMATION_LEARNING_RATES
    ]


def test_confirmation_grid_and_lanes_are_balanced() -> None:
    jobs = confirmation_jobs()
    validate_confirmation_jobs(jobs)
    lanes = confirmation_lanes(jobs)
    assert [len(lane) for lane in lanes] == [3, 3]
    assert sorted(job["job_name"] for lane in lanes for job in lane) == sorted(
        job["job_name"] for job in jobs
    )
    assert all({job["learning_rate"] for job in lane} == {5e-5, 1e-4} for lane in lanes)
    with pytest.raises(ValueError, match="exact two-rate"):
        validate_confirmation_jobs(jobs[:-1])


def test_confirmation_training_command_uses_scheduled_muon() -> None:
    command = training_command(confirmation_job(2, 1e-4), distributed_processes=1)
    assert "student.training.optimizer=muon" in command
    assert "student.training.learning_rate=0.0001" in command
    assert "student.training.lr_scheduler_type=linear" in command
    assert "student.training.muon_adjust_lr_fn=match_rms_adamw" in command
    assert not any("selection_manifest" in value for value in command)


def synthetic_results(muon_gain: float = 0.003) -> pd.DataFrame:
    rows = []
    for seed in CONFIRMATION_SEEDS:
        baseline = 0.96 + 0.001 * seed
        for optimizer, learning_rate, gain in (
            ("adamw", 5e-5, 0.0),
            ("muon", 5e-5, muon_gain),
            ("muon", 1e-4, muon_gain - 0.001),
        ):
            row = {
                "optimizer": optimizer,
                "learning_rate": learning_rate,
                "seed": seed,
                "macro_auroc": baseline + gain,
                "macro_balanced_accuracy": 0.91 + gain / 2,
                "macro_brier": 0.07 - gain / 2,
                "pooled_auroc": baseline + gain,
                "pooled_balanced_accuracy": 0.91 + gain / 2,
                "pooled_brier": 0.07 - gain / 2,
                "unique_scores": 800,
                "tied_rows": 22,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def test_paired_contrasts_and_selection_gate() -> None:
    frame = synthetic_results()
    contrasts = paired_confirmation_contrasts(frame)
    assert len(contrasts) == 6
    assert set(METRICS).issubset(
        column.removeprefix("muon_minus_adamw_")
        for column in contrasts.columns
        if column.startswith("muon_minus_adamw_")
    )
    aggregate = (
        frame.groupby(["optimizer", "learning_rate"])
        .agg(
            seeds=("seed", "count"),
            macro_auroc_mean=("macro_auroc", "mean"),
            macro_auroc_std=("macro_auroc", "std"),
            macro_balanced_accuracy_mean=("macro_balanced_accuracy", "mean"),
            macro_balanced_accuracy_std=("macro_balanced_accuracy", "std"),
            macro_brier_mean=("macro_brier", "mean"),
            macro_brier_std=("macro_brier", "std"),
            pooled_auroc_mean=("pooled_auroc", "mean"),
            pooled_auroc_std=("pooled_auroc", "std"),
            unique_scores_mean=("unique_scores", "mean"),
            tied_rows_mean=("tied_rows", "mean"),
        )
        .reset_index()
    )
    summary = selection_summary(aggregate, contrasts)
    assert summary["selected_recipe"]["optimizer"] == "muon"
    assert summary["selected_recipe"]["learning_rate"] == pytest.approx(5e-5)
