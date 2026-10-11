"""Portable logistic combinations of behavior and injection logit margins."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def feature_matrix(rows: list[dict], arm: str) -> np.ndarray:
    """Use model outputs only; never include labels or template metadata."""
    safety = np.array([r["safety_margin"] for r in rows], dtype=float)
    injection = np.array([r["injection_margin"] for r in rows], dtype=float)
    if arm == "safety_calibrated":
        values = safety[:, None]
    elif arm == "two_score":
        values = np.column_stack([safety, injection])
    elif arm == "interaction":
        values = np.column_stack([safety, injection, safety * injection])
    else:
        raise ValueError(f"unknown fusion arm: {arm}")
    if not len(rows) or not np.isfinite(values).all():
        raise ValueError("empty or nonfinite fusion features")
    return values


def fit_combiner(rows: list[dict], arm: str) -> dict:
    """Fit one fixed regularized model, export its complete numeric interface."""
    values = feature_matrix(rows, arm)
    labels = np.array([r["ground_truth"] for r in rows])
    if set(labels) != {0, 1}:
        raise ValueError("fusion fitting requires both behavior labels")
    scaler = StandardScaler().fit(values)
    model = LogisticRegression(C=1, max_iter=1000, random_state=0).fit(
        scaler.transform(values), labels
    )
    if model.n_iter_[0] >= 1000:
        raise ValueError("fusion fitting did not converge")
    return {
        "arm": arm,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficient": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "iterations": int(model.n_iter_[0]),
        "classes": model.classes_.tolist(),
    }


def combined_scores(rows: list[dict], model: dict) -> np.ndarray:
    """Apply exported coefficients with a numerically stable sigmoid."""
    values = feature_matrix(rows, model["arm"])
    margin = ((values - model["mean"]) / model["scale"]) @ np.array(
        model["coefficient"]
    ) + model["intercept"]
    exp = np.exp(-np.abs(margin))
    return np.where(margin >= 0, 1 / (1 + exp), exp / (1 + exp))


def fit_threshold(rows: list[dict], scores: np.ndarray) -> float:
    """Empirical strict > threshold permits at most 2% fitting-negative alarms."""
    negatives = sorted(
        float(score)
        for row, score in zip(rows, scores, strict=True)
        if row["ground_truth"] == 0
    )
    if not negatives or not np.isfinite(scores).all():
        raise ValueError("invalid threshold fitting population")
    return negatives[len(negatives) - int(0.02 * len(negatives)) - 1]
