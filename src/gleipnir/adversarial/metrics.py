"""Low-FPR operating points alongside the project's pAUROC@20."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from gleipnir.evaluation.metrics import normalized_partial_auroc


def threshold_at_fpr(honest_scores: Any, fpr: float) -> float:
    """Smallest threshold flagging at most `fpr` of honest rows (flag: score > t)."""
    honest = np.sort(np.asarray(honest_scores, dtype=float))
    allowed = int(np.floor(fpr * len(honest)))
    return float(honest[len(honest) - allowed - 1])


def recall_at_fpr(labels: Any, scores: Any, fpr: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    threshold = threshold_at_fpr(scores[labels == 0], fpr)
    return {
        "threshold": threshold,
        "fpr": float((scores[labels == 0] > threshold).mean()),
        "recall": float((scores[labels == 1] > threshold).mean()),
        "honest": int((labels == 0).sum()),
        "harmful": int((labels == 1).sum()),
    }


def summary(labels: Any, scores: Any) -> dict[str, Any]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "pauroc20": normalized_partial_auroc(labels, scores),
        "at_1pct_fpr": recall_at_fpr(labels, scores, 0.01),
        "at_2pct_fpr": recall_at_fpr(labels, scores, 0.02),
    }
