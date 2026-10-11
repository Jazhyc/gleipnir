"""Label-conditioned removal of confident teacher disagreements."""

from __future__ import annotations

import math


def remove_disagreement(label: int, score: float, delta: float) -> bool:
    """Preserve the uncertainty band; at delta zero classify ties as harmful."""
    if label not in (0, 1) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("invalid binary label or teacher probability")
    if not math.isfinite(delta) or not 0 <= delta < 0.5:
        raise ValueError("delta must be finite and in [0, 0.5)")
    if delta == 0:
        return int(score >= 0.5) != label
    return (label == 1 and score < 0.5 - delta) or (label == 0 and score > 0.5 + delta)
