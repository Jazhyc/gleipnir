"""Pure helpers for paired optimizer learning-rate experiments."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

DEFAULT_LEARNING_RATES = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4)
DEFAULT_OPTIMIZERS = ("adamw", "muon")


def learning_rate_tag(learning_rate: float) -> str:
    """Return a stable filesystem-safe scientific-notation tag."""
    value = Decimal(str(learning_rate)).normalize()
    if not value.is_finite() or value <= 0:
        raise ValueError("learning rate must be finite and positive")
    scientific = f"{float(value):.0e}"
    return scientific.replace("+", "p").replace("-", "m")


def validate_learning_rates(values: list[float] | tuple[float, ...]) -> list[float]:
    selected = sorted(set(float(value) for value in values))
    if not selected or any(value <= 0 or value > 1e-2 for value in selected):
        raise ValueError("learning rates must be unique values in (0, 1e-2]")
    return selected


def paired_optimizer_lanes(
    jobs: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Put AdamW and Muon on separate lanes in identical LR order."""
    by_optimizer = {
        optimizer: sorted(
            (job for job in jobs if job["optimizer"] == optimizer),
            key=lambda job: (
                abs(float(job["learning_rate"]) - 5e-5),
                float(job["learning_rate"]),
            ),
        )
        for optimizer in DEFAULT_OPTIMIZERS
    }
    if any(
        len(lane) * len(DEFAULT_OPTIMIZERS) != len(jobs)
        for lane in by_optimizer.values()
    ):
        raise ValueError("jobs must contain a complete paired AdamW/Muon LR grid")
    adamw_rates = [float(job["learning_rate"]) for job in by_optimizer["adamw"]]
    muon_rates = [float(job["learning_rate"]) for job in by_optimizer["muon"]]
    if adamw_rates != muon_rates:
        raise ValueError("AdamW and Muon learning rates are not paired")
    return [by_optimizer["adamw"], by_optimizer["muon"]]
