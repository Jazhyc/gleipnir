"""Evaluation helpers for continuous monitor scores."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


def normalized_partial_auroc(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    max_fpr: float = 0.2,
) -> float:
    """Return raw ROC area through ``max_fpr``, normalized by that FPR width.

    This is the monitoring-paper definition of pAUROC@20. It deliberately does
    not use scikit-learn's standardized partial-AUC correction.
    """
    if not 0.0 < max_fpr <= 1.0:
        raise ValueError("max_fpr must lie in (0, 1]")
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("partial AUROC requires both binary labels")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")

    fpr, tpr, _ = roc_curve(labels, scores)
    stop = int(np.searchsorted(fpr, max_fpr, side="right"))
    if stop < len(fpr):
        clipped_fpr = np.append(fpr[:stop], max_fpr)
        clipped_tpr = np.append(
            tpr[:stop],
            np.interp(max_fpr, fpr[stop - 1 : stop + 1], tpr[stop - 1 : stop + 1]),
        )
    else:
        clipped_fpr = fpr
        clipped_tpr = tpr
    return float(np.trapezoid(clipped_tpr, clipped_fpr) / max_fpr)


def evaluate_binary_monitor(
    frame: pd.DataFrame,
    *,
    group_column: str = "dataset",
    label_column: str = "label",
    score_column: str = "score",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Report per-group and macro ranking, calibration, and threshold metrics."""
    required = {group_column, label_column, score_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"evaluation frame is missing columns: {sorted(missing)}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    scores = frame[score_column].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be finite and lie in [0, 1]")

    rows = []
    for group, subset in frame.groupby(group_column, sort=True):
        labels = subset[label_column].to_numpy(dtype=int)
        group_scores = subset[score_column].to_numpy(dtype=float)
        if not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError(f"group {group!r} contains non-binary labels")
        predictions = group_scores >= threshold
        auroc = (
            float(roc_auc_score(labels, group_scores))
            if len(np.unique(labels)) == 2
            else None
        )
        pauroc_at_20 = (
            normalized_partial_auroc(labels, group_scores)
            if len(np.unique(labels)) == 2
            else None
        )
        positives = labels == 1
        negatives = labels == 0
        recall = float(predictions[positives].mean()) if positives.any() else None
        fpr = float(predictions[negatives].mean()) if negatives.any() else None
        rows.append(
            {
                "group": str(group),
                "n": len(subset),
                "auroc": auroc,
                "pauroc_at_20": pauroc_at_20,
                "brier": float(brier_score_loss(labels, group_scores)),
                "balanced_accuracy": (
                    float(balanced_accuracy_score(labels, predictions))
                    if len(np.unique(labels)) == 2
                    else None
                ),
                "recall": recall,
                "fpr": fpr,
                "unique_scores": int(len(np.unique(group_scores))),
            }
        )

    metric_names = (
        "auroc",
        "pauroc_at_20",
        "brier",
        "balanced_accuracy",
        "recall",
        "fpr",
    )
    macro = {
        name: (
            float(np.mean([row[name] for row in rows if row[name] is not None]))
            if any(row[name] is not None for row in rows)
            else None
        )
        for name in metric_names
    }
    return {
        "n": len(frame),
        "threshold": threshold,
        "partial_auroc_max_fpr": 0.2,
        "macro": macro,
        "groups": rows,
        "diagnostics": {
            "unique_scores": int(len(np.unique(scores))),
            "tied_rows": int(len(scores) - len(np.unique(scores))),
        },
    }
