"""Descriptive binary-probability reliability diagnostics."""

import numpy as np


def binary_calibration(labels, probabilities, n_bins: int = 10) -> dict:
    """Equal-width positive-class ECE; empty bins do not contribute.

    Wilson intervals describe bin event rates under independent Bernoulli
    sampling, not uncertainty in model predictions or grouped-population error.
    Log loss clips probabilities at 1e-15 only for numerical evaluation.
    """
    y, p = np.asarray(labels, dtype=float), np.asarray(probabilities, dtype=float)
    if (
        y.ndim != 1
        or y.shape != p.shape
        or not y.size
        or n_bins < 1
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("Expected paired binary labels and finite probabilities")
    edges = np.linspace(0, 1, n_bins + 1)
    assignments = np.minimum(np.searchsorted(edges, p, side="right") - 1, n_bins - 1)
    bins = []
    for i in range(n_bins):
        mask = assignments == i
        n = int(mask.sum())
        record = dict(lower=float(edges[i]), upper=float(edges[i + 1]), n=n)
        if n:
            observed = float(y[mask].mean())
            z = 1.959963984540054
            denominator = 1 + z * z / n
            center = (observed + z * z / (2 * n)) / denominator
            half = (
                z
                * np.sqrt(observed * (1 - observed) / n + z * z / (4 * n * n))
                / denominator
            )
            record.update(
                mean_probability=float(p[mask].mean()),
                observed_rate=observed,
                wilson95=[max(0.0, center - half), min(1.0, center + half)],
            )
        bins.append(record)
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    confidence = np.maximum(p, 1 - p)
    accuracy = float(np.mean((p >= 0.5) == y))
    return dict(
        n=len(y),
        prevalence=float(y.mean()),
        mean_probability=float(p.mean()),
        probability_bias=float((p - y).mean()),
        brier=float(np.mean((p - y) ** 2)),
        log_loss=float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        ece=sum(
            b["n"] / len(y) * abs(b["mean_probability"] - b["observed_rate"])
            for b in bins
            if b["n"]
        ),
        accuracy=accuracy,
        mean_confidence=float(confidence.mean()),
        confidence_minus_accuracy=float(confidence.mean() - accuracy),
        bins=bins,
    )
