"""Stable Bernoulli teacher-fit metrics from raw decision margins."""

import math


def row_fit(q: float, margin: float) -> dict[str, float]:
    """Return soft CE, teacher entropy and forward KL in nats."""
    if not 0 <= q <= 1 or not math.isfinite(margin):
        raise ValueError("invalid teacher target or student margin")
    ce = (1 - q) * max(margin, 0) + q * max(-margin, 0)
    ce += math.log1p(math.exp(-abs(margin)))
    entropy = -sum(p * math.log(p) for p in (q, 1 - q) if p > 0)
    return {"ce": ce, "teacher_entropy": entropy, "kl": max(ce - entropy, 0)}
