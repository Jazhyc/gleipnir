"""Pure helpers for the joint data-volume and adapter-rank sweep."""

from __future__ import annotations

from typing import Any

DEFAULT_FRACTIONS = (0.10, 0.25, 0.50)
FULL_FRACTION = 1.0
DEFAULT_RANKS = (4, 16, 64, 256)
DEFAULT_SEEDS = (0, 1, 2)


def fraction_tag(fraction: float) -> str:
    """Return the fixed three-digit percentage tag used in job identities."""
    if fraction <= 0 or fraction > 1:
        raise ValueError("fraction must be in (0, 1]")
    return f"f{round(fraction * 100):03d}"


def validate_design(
    fractions: list[float] | tuple[float, ...],
    ranks: list[int] | tuple[int, ...],
    seeds: list[int] | tuple[int, ...],
) -> tuple[list[float], list[int], list[int]]:
    """Validate and canonicalize the predeclared interaction grid."""
    selected_fractions = sorted(set(float(value) for value in fractions))
    selected_ranks = sorted(set(int(value) for value in ranks))
    selected_seeds = sorted(set(int(value) for value in seeds))
    if not selected_fractions or any(
        value <= 0 or value >= FULL_FRACTION for value in selected_fractions
    ):
        raise ValueError("new training fractions must be unique values in (0, 1)")
    if not selected_ranks or any(
        value not in DEFAULT_RANKS for value in selected_ranks
    ):
        raise ValueError(f"ranks must be selected from {list(DEFAULT_RANKS)}")
    if not selected_seeds:
        raise ValueError("at least one seed is required")
    return selected_fractions, selected_ranks, selected_seeds


def balanced_interaction_lanes(
    jobs: list[dict[str, Any]], lane_count: int
) -> list[list[dict[str, Any]]]:
    """Balance jobs by rows with a small measured high-rank cost correction."""
    if lane_count < 1:
        raise ValueError("lane_count must be positive")
    lanes: list[list[dict[str, Any]]] = [[] for _ in range(lane_count)]
    loads = [0.0] * lane_count
    ordered = sorted(
        jobs,
        key=lambda job: (
            -int(job["train_rows"]) * (1.0 + int(job["rank"]) / 8192.0),
            str(job["job_name"]),
        ),
    )
    for job in ordered:
        lane = min(range(lane_count), key=lambda index: (loads[index], index))
        lanes[lane].append(job)
        loads[lane] += int(job["train_rows"]) * (
            1.0 + int(job["rank"]) / 8192.0
        )
    return lanes
