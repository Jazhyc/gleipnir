"""Pure contracts for the Kimi-soft tool-trajectory scaling campaign."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

PAPER_SCHEDULE = {
    204: 378,
    504: 375,
    996: 372,
    2_004: 372,
    4_008: 750,
    8_688: 813,
}
CAMPAIGN_SEED = 0
LORA_RANK = 128
LORA_ALPHA = 256
EFFECTIVE_BATCH_SIZE = 32
MAX_LENGTH = 29_696


def stable_row_order(rows: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Return a seed-stable order independent of source artifact row order."""
    decorated = []
    for row in rows:
        identity = str(row["index"])
        digest = hashlib.sha256(f"{seed}\0{identity}".encode()).digest()
        decorated.append((digest, identity, row))
    decorated.sort(key=lambda value: (value[0], value[1]))
    return [row for _, _, row in decorated]


def equal_capped_allocation(
    capacities: dict[tuple[str, int], int], total: int
) -> dict[tuple[str, int], int]:
    """Allocate exactly ``total`` rows as equally as stratum capacities permit."""
    if total < 1 or total > sum(capacities.values()):
        raise ValueError("requested rows must lie within the available pool")
    if not capacities or any(value < 1 for value in capacities.values()):
        raise ValueError("every source-label stratum must be non-empty")
    keys = sorted(capacities)
    allocation = {key: 0 for key in keys}
    remaining = total
    active = list(keys)
    while remaining:
        share = remaining / len(active)
        base = max(1, math.floor(share))
        progressed = False
        for key in list(active):
            available = capacities[key] - allocation[key]
            take = min(base, available, remaining)
            if take:
                allocation[key] += take
                remaining -= take
                progressed = True
            if allocation[key] == capacities[key]:
                active.remove(key)
            if not remaining:
                break
        if not progressed:
            raise AssertionError("capped allocation made no progress")
    return allocation


def nested_count_selections(
    rows: list[dict[str, Any]],
    counts: Iterable[int],
    seed: int,
) -> dict[int, list[dict[str, Any]]]:
    """Build exact, nested, approximately source-class-balanced selections."""
    requested = sorted(set(int(value) for value in counts))
    if not requested or requested[0] < 1 or requested[-1] > len(rows):
        raise ValueError("selection counts must be unique values within the pool")
    identities = [str(row["index"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("training rows contain duplicate identities")

    strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"non-binary label for {row['index']!r}")
        strata[(str(row["source_dataset"]), label)].append(row)
    ordered = {
        key: stable_row_order(values, seed)
        for key, values in strata.items()
    }
    capacities = {key: len(values) for key, values in ordered.items()}

    selections: dict[int, list[dict[str, Any]]] = {}
    previous: set[str] = set()
    for count in requested:
        allocation = (
            capacities.copy()
            if count == len(rows)
            else equal_capped_allocation(capacities, count)
        )
        selected = [
            row
            for key in sorted(ordered)
            for row in ordered[key][: allocation[key]]
        ]
        selected.sort(key=lambda row: int(row["source_row_index"]))
        selected_ids = {str(row["index"]) for row in selected}
        if len(selected) != count or not previous.issubset(selected_ids):
            raise AssertionError("selection is not exact and nested")
        selections[count] = selected
        previous = selected_ids
    return selections


def scaling_job_name(rows: int) -> str:
    return f"soft-n{rows:05d}-seed{CAMPAIGN_SEED}"


def validate_jobs(jobs: list[dict[str, Any]]) -> None:
    """Fail closed if the frozen seven-job soft-only design drifts."""
    expected_names = {scaling_job_name(rows) for rows in PAPER_SCHEDULE} | {
        f"soft-n21837-mixed-seed{CAMPAIGN_SEED}"
    }
    names = {str(job["job_name"]) for job in jobs}
    if names != expected_names or len(jobs) != 7:
        raise ValueError(f"expected frozen seven-job design, got {sorted(names)}")
    for job in jobs:
        if job.get("target") != "kimi_soft":
            raise ValueError("this campaign permits Kimi soft targets only")
        if int(job["rank"]) != LORA_RANK or int(job["lora_alpha"]) != LORA_ALPHA:
            raise ValueError("adapter rank contract drifted")
        if int(job["effective_batch_size"]) != EFFECTIVE_BATCH_SIZE:
            raise ValueError("effective batch contract drifted")
        if int(job["max_length"]) != MAX_LENGTH:
            raise ValueError("maximum sequence length contract drifted")
