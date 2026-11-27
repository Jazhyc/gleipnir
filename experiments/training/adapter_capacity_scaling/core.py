"""Pure helpers for the adapter-capacity sweep."""

from __future__ import annotations

from typing import Any

DEFAULT_RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_SEEDS = (0, 1, 2)
QLORA_MICRO_BATCH_SIZE = 8
QLORA_GRADIENT_ACCUMULATION_STEPS = 4
QLORA_EFFECTIVE_BATCH_SIZE = (
    QLORA_MICRO_BATCH_SIZE * QLORA_GRADIENT_ACCUMULATION_STEPS
)


def validate_ranks(ranks: list[int] | tuple[int, ...]) -> list[int]:
    selected = sorted(set(int(rank) for rank in ranks))
    if not selected or any(rank < 1 or rank > 256 for rank in selected):
        raise ValueError("LoRA ranks must be unique values in [1, 256]")
    if any(rank & (rank - 1) for rank in selected):
        raise ValueError("LoRA ranks must be powers of two")
    return selected


def split_jobs(
    jobs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lora = [job for job in jobs if job["capacity_kind"] == "lora"]
    full = [job for job in jobs if job["capacity_kind"] == "full"]
    if len(lora) + len(full) != len(jobs):
        raise ValueError("jobs contain an unknown capacity_kind")
    return lora, full


def balanced_lora_lanes(
    jobs: list[dict[str, Any]], lane_count: int
) -> list[list[dict[str, Any]]]:
    """Place larger ranks first using rank as a small runtime-cost correction."""
    if lane_count < 1:
        raise ValueError("lane_count must be positive")
    if any(job["capacity_kind"] != "lora" for job in jobs):
        raise ValueError("balanced_lora_lanes accepts only LoRA jobs")
    lanes: list[list[dict[str, Any]]] = [[] for _ in range(lane_count)]
    loads = [0] * lane_count
    for job in sorted(jobs, key=lambda item: (-int(item["rank"]), item["job_name"])):
        lane = min(range(lane_count), key=lambda index: (loads[index], index))
        lanes[lane].append(job)
        loads[lane] += 256 + int(job["rank"])
    return lanes
