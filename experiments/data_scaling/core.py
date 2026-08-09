"""Pure helpers for reproducible nested data-scaling sweeps."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

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


def fit_bounded_power_law(
    sizes: Iterable[int],
    values: Iterable[float],
    *,
    grid_points: int = 2000,
) -> dict[str, float]:
    """Fit metric(N) = asymptote - coefficient * N**(-exponent).

    The asymptote is searched on a dense deterministic grid. For each candidate,
    the remaining two parameters are obtained by linear regression in log space,
    and candidates are compared by squared error in the original metric space.
    """
    x = np.asarray(list(sizes), dtype=float)
    y = np.asarray(list(values), dtype=float)
    if len(x) < 3 or x.shape != y.shape:
        raise ValueError("at least three paired sizes and values are required")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (x <= 0).any():
        raise ValueError("sizes and values must be finite, with positive sizes")
    if (y < 0).any() or (y > 1).any():
        raise ValueError("metric values must lie in [0, 1]")

    lower = min(1.0, float(y.max()) + 1.0e-6)
    if lower >= 1.0:
        lower = float(y.max())
    candidates = np.linspace(lower, 1.0, grid_points)
    best: tuple[float, float, float, float] | None = None
    log_x = np.log(x)
    design = np.column_stack([np.ones_like(log_x), log_x])
    for asymptote in candidates:
        residual = asymptote - y
        if (residual <= 0).any():
            continue
        intercept, slope = np.linalg.lstsq(
            design, np.log(residual), rcond=None
        )[0]
        exponent = -float(slope)
        coefficient = math.exp(float(intercept))
        if exponent <= 0 or not math.isfinite(coefficient):
            continue
        predicted = asymptote - coefficient * x ** (-exponent)
        squared_error = float(np.square(y - predicted).sum())
        candidate = (squared_error, float(asymptote), coefficient, exponent)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("could not fit a positive-exponent bounded power law")
    squared_error, asymptote, coefficient, exponent = best
    return {
        "asymptote": asymptote,
        "coefficient": coefficient,
        "exponent": exponent,
        "rmse": math.sqrt(squared_error / len(x)),
    }
