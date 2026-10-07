import pandas as pd
import pytest

from experiments.adapter_capacity_scaling.run_train import training_command
from experiments.optimization_regularization_screen.core import (
    BASELINE_JOB_NAME,
    balanced_screen_lanes,
    screen_variants,
    validate_screen_jobs,
)
from experiments.optimization_regularization_screen.summarize import (
    contrast_frame,
    summary_payload,
)


def screen_job(variant: dict[str, object]) -> dict[str, object]:
    job = {
        "job_name": variant["job_name"],
        "intervention_family": variant["intervention_family"],
        "capacity_kind": "lora",
        "seed": 0,
        "rank": 128,
        "lora_alpha": 256,
        "optimizer": "adamw",
        "learning_rate": 5e-5,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.03,
        "num_train_epochs": 2.0,
        "lora_dropout": 0.0,
        "weight_decay": 0.0,
        "dataset_sampling": "proportional",
        "soft_target_logit_scale": 1.0,
        "save_strategy": "steps",
        "save_steps": 88,
        "save_total_limit": 6,
        "save_only_model": True,
        "train_rows": 5589,
        "selection_manifest": "/data/selection.jsonl",
        "student_rows": "/data/student.jsonl",
        "soft_targets": "/data/soft.jsonl",
        "validation": "/data/development.jsonl",
        "output_dir": "/results/run",
        "causal_adapter_dir": "/results/run/causal_adapter",
        "model_dir": "/results/run/model",
    }
    job.update(variant)
    return job


def screen_jobs() -> list[dict[str, object]]:
    return [screen_job(variant) for variant in screen_variants()]


def test_frozen_screen_has_sixteen_balanced_cells() -> None:
    jobs = screen_jobs()
    validate_screen_jobs(jobs)
    assert len(jobs) == 16
    assert sum(job["job_name"] == BASELINE_JOB_NAME for job in jobs) == 1
    lanes = balanced_screen_lanes(jobs)
    assert sorted(job["job_name"] for lane in lanes for job in lane) == sorted(
        job["job_name"] for job in jobs
    )
    loads = [
        sum(job["train_rows"] * job["num_train_epochs"] for job in lane)
        for lane in lanes
    ]
    assert loads[0] == loads[1]
    with pytest.raises(ValueError, match="frozen 16-cell"):
        validate_screen_jobs(jobs[:-1])


def test_training_command_carries_screen_overrides() -> None:
    job = screen_job(
        {
            "job_name": "combined-test",
            "intervention_family": "test",
            "lora_dropout": 0.05,
            "dataset_sampling": "uniform_dataset",
            "soft_target_logit_scale": 1.5,
            "lr_scheduler_type": "cosine",
        }
    )
    command = training_command(job, distributed_processes=1)
    expected = {
        "student.lora.dropout=0.05",
        "student.training.dataset_sampling=uniform_dataset",
        "student.training.soft_target_logit_scale=1.5",
        "student.training.lr_scheduler_type=cosine",
        "student.training.save_strategy=steps",
        "student.training.save_steps=88",
        "student.training.save_total_limit=6",
        "student.training.save_only_model=True",
    }
    assert expected.issubset(command)


def synthetic_results() -> pd.DataFrame:
    rows = []
    for job_name, family, auroc, ba, brier in (
        (BASELINE_JOB_NAME, "baseline", 0.94, 0.90, 0.10),
        ("eligible", "dropout", 0.95, 0.899, 0.101),
        ("bad-ba", "schedule", 0.96, 0.897, 0.10),
        ("bad-brier", "sampling", 0.97, 0.90, 0.103),
    ):
        rows.append(
            {
                "job_name": job_name,
                "intervention_family": family,
                "macro_auroc": auroc,
                "macro_balanced_accuracy": ba,
                "macro_brier": brier,
                "pooled_auroc": auroc,
                "pooled_balanced_accuracy": ba,
                "pooled_brier": brier,
            }
        )
    return pd.DataFrame(rows)


def test_summary_applies_metric_constraints_before_shortlisting() -> None:
    frame = synthetic_results()
    contrasts = contrast_frame(frame)
    eligibility = contrasts.set_index("job_name")["eligible"].to_dict()
    assert eligibility == {
        BASELINE_JOB_NAME: True,
        "eligible": True,
        "bad-ba": False,
        "bad-brier": False,
    }
    summary = summary_payload(frame, contrasts)
    assert [row["job_name"] for row in summary["replication_shortlist"]] == [
        "eligible"
    ]
    assert summary["promotion_permitted"] is False
