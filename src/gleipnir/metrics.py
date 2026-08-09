"""Evaluation helpers for continuous monitor scores."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score


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
        positives = labels == 1
        negatives = labels == 0
        recall = float(predictions[positives].mean()) if positives.any() else None
        fpr = float(predictions[negatives].mean()) if negatives.any() else None
        rows.append(
            {
                "group": str(group),
                "n": len(subset),
                "auroc": auroc,
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

    metric_names = ("auroc", "brier", "balanced_accuracy", "recall", "fpr")
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
        "macro": macro,
        "groups": rows,
        "diagnostics": {
            "unique_scores": int(len(np.unique(scores))),
            "tied_rows": int(len(scores) - len(np.unique(scores))),
        },
    }
