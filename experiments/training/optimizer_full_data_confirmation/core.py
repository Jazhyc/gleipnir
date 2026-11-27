"""Pure design and scheduling helpers for full-data optimizer confirmation."""

from __future__ import annotations

from typing import Any

CONFIRMATION_LEARNING_RATES = (5e-5, 1e-4)
CONFIRMATION_SEEDS = (0, 1, 2)
ADAMW_REFERENCE_LEARNING_RATE = 5e-5


def validate_confirmation_jobs(jobs: list[dict[str, Any]]) -> None:
    """Require the exact two-rate by three-seed Muon confirmation grid."""
    expected = {
        (seed, learning_rate)
        for seed in CONFIRMATION_SEEDS
        for learning_rate in CONFIRMATION_LEARNING_RATES
    }
    observed = {
        (int(job["seed"]), float(job["learning_rate"])) for job in jobs
    }
    if observed != expected or len(jobs) != len(expected):
        raise ValueError("jobs must contain the exact two-rate by three-seed grid")
    if any(job.get("optimizer") != "muon" for job in jobs):
        raise ValueError("confirmation training jobs must all use Muon")
    if any(int(job.get("rank", 0)) != 64 for job in jobs):
        raise ValueError("confirmation training jobs must all use rank 64")


def confirmation_lanes(jobs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Balance seeds and rates across two equal-runtime H100 lanes."""
    validate_confirmation_jobs(jobs)
    lanes: list[list[dict[str, Any]]] = [[], []]
    rate_index = {
        learning_rate: index
        for index, learning_rate in enumerate(CONFIRMATION_LEARNING_RATES)
    }
    ordered = sorted(
        jobs, key=lambda item: (int(item["seed"]), item["learning_rate"])
    )
    for job in ordered:
        lane = (int(job["seed"]) + rate_index[float(job["learning_rate"])]) % 2
        lanes[lane].append(job)
    if any(len(lane) != 3 for lane in lanes):
        raise AssertionError("confirmation lanes must contain three jobs each")
    return lanes
