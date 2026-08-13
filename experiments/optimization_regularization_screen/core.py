"""Pure design and scheduling helpers for the optimization screen."""

from __future__ import annotations

from typing import Any

BASELINE_JOB_NAME = "baseline"


def screen_variants() -> list[dict[str, Any]]:
    """Return the frozen baseline plus fifteen single-factor interventions."""
    return [
        {"job_name": BASELINE_JOB_NAME, "intervention_family": "baseline"},
        {
            "job_name": "warmup-000",
            "intervention_family": "warmup",
            "warmup_ratio": 0.0,
        },
        {
            "job_name": "warmup-010",
            "intervention_family": "warmup",
            "warmup_ratio": 0.10,
        },
        {
            "job_name": "schedule-cosine",
            "intervention_family": "schedule",
            "lr_scheduler_type": "cosine",
        },
        {
            "job_name": "schedule-constant-matched",
            "intervention_family": "schedule",
            "lr_scheduler_type": "constant_with_warmup",
            "learning_rate": 2.5e-5,
        },
        {
            "job_name": "epochs-1",
            "intervention_family": "training_horizon",
            "num_train_epochs": 1.0,
        },
        {
            "job_name": "epochs-3",
            "intervention_family": "training_horizon",
            "num_train_epochs": 3.0,
        },
        {
            "job_name": "dropout-0025",
            "intervention_family": "lora_dropout",
            "lora_dropout": 0.025,
        },
        {
            "job_name": "dropout-0050",
            "intervention_family": "lora_dropout",
            "lora_dropout": 0.05,
        },
        {
            "job_name": "dropout-0100",
            "intervention_family": "lora_dropout",
            "lora_dropout": 0.10,
        },
        {
            "job_name": "weight-decay-001",
            "intervention_family": "weight_decay",
            "weight_decay": 0.01,
        },
        {
            "job_name": "weight-decay-010",
            "intervention_family": "weight_decay",
            "weight_decay": 0.10,
        },
        {
            "job_name": "sampling-sqrt-balanced",
            "intervention_family": "dataset_sampling",
            "dataset_sampling": "sqrt_balanced",
        },
        {
            "job_name": "sampling-uniform-dataset",
            "intervention_family": "dataset_sampling",
            "dataset_sampling": "uniform_dataset",
        },
        {
            "job_name": "target-scale-15",
            "intervention_family": "target_smoothing",
            "soft_target_logit_scale": 1.5,
        },
        {
            "job_name": "target-scale-20",
            "intervention_family": "target_smoothing",
            "soft_target_logit_scale": 2.0,
        },
    ]


def validate_screen_jobs(jobs: list[dict[str, Any]]) -> None:
    """Require exactly one job for every frozen screen variant."""
    expected = {variant["job_name"] for variant in screen_variants()}
    observed = [str(job["job_name"]) for job in jobs]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("jobs do not match the frozen 16-cell screen")
    if any(job.get("optimizer") != "adamw" for job in jobs):
        raise ValueError("all screen jobs must use AdamW")
    if any(int(job.get("rank", 0)) != 128 for job in jobs):
        raise ValueError("all screen jobs must use rank 128")


def balanced_screen_lanes(
    jobs: list[dict[str, Any]], lane_count: int = 2
) -> list[list[dict[str, Any]]]:
    """LPT-schedule cells using training rows times epochs as runtime cost."""
    validate_screen_jobs(jobs)
    if lane_count < 1:
        raise ValueError("lane_count must be positive")
    lanes: list[list[dict[str, Any]]] = [[] for _ in range(lane_count)]
    loads = [0.0] * lane_count
    ordered = sorted(
        jobs,
        key=lambda job: (
            -float(job["train_rows"]) * float(job["num_train_epochs"]),
            str(job["job_name"]),
        ),
    )
    for job in ordered:
        lane = min(range(lane_count), key=lambda index: (loads[index], index))
        lanes[lane].append(job)
        loads[lane] += float(job["train_rows"]) * float(job["num_train_epochs"])
    return lanes
