"""Reusable fitting helpers for empirical scaling experiments."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def fit_bounded_power_law(
    sizes: Iterable[int | float],
    values: Iterable[float],
    *,
    grid_points: int = 2000,
) -> dict[str, float]:
    """Fit ``metric(x) = asymptote - coefficient * x**(-exponent)``."""
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
