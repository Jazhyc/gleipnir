"""Pure helpers for reproducible nested data-scaling sweeps."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from gleipnir.scaling import fit_bounded_power_law

__all__ = ["fit_bounded_power_law", "nested_stratified_selections", "record_identity"]

Record = dict[str, Any]
Identity = tuple[str, Any]


def record_identity(record: Record) -> Identity:
    """Return the stable dataset/index identity used by teacher caches."""
    return str(record["dataset"]), record["index"]


def nested_stratified_selections(
    records: list[Record],
    fractions: Iterable[float],
    seed: int,
) -> dict[float, list[Record]]:
    """Select nested prefixes of a stable hash order in each dataset/label stratum."""
    ordered_fractions = sorted(set(float(value) for value in fractions))
    if not ordered_fractions or ordered_fractions[0] <= 0 or ordered_fractions[-1] > 1:
        raise ValueError("fractions must be unique values in (0, 1]")

    identities = [record_identity(record) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("training records contain duplicate dataset/index identities")

    strata: dict[tuple[str, int], list[tuple[bytes, Record]]] = defaultdict(list)
    for record in records:
        dataset, index = record_identity(record)
        label = int(record["label"])
        if label not in (0, 1):
            raise ValueError(f"non-binary label for {(dataset, index)!r}: {label}")
        digest = hashlib.sha256(
            f"{seed}\0{dataset}\0{label}\0{index}".encode()
        ).digest()
        strata[(dataset, label)].append((digest, record))
    for candidates in strata.values():
        candidates.sort(key=lambda item: item[0])

    selections: dict[float, list[Record]] = {}
    previous: set[Identity] = set()
    for fraction in ordered_fractions:
        selected: set[Identity] = set()
        for candidates in strata.values():
            count = max(1, int(len(candidates) * fraction + 0.5))
            selected.update(record_identity(record) for _, record in candidates[:count])
        if not previous.issubset(selected):
            raise AssertionError("fraction selections are not nested")
        selections[fraction] = [
            record for record in records if record_identity(record) in selected
        ]
        previous = selected
    return selections
