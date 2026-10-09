"""Describe fixed-axis training associations without claiming example influence."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def paired_values(base: dict, trained: dict) -> dict:
    """Compare identical prompts in the unchanged residual coordinate system."""
    if (base["id"], base["prompt_sha256"], base["prompt_tokens"]) != (
        trained["id"],
        trained["prompt_sha256"],
        trained["prompt_tokens"],
    ):
        raise ValueError("model comparison identity drift")
    result = {
        k: v for k, v in trained.items() if k not in ["logits", "latency_seconds"]
    }
    for layer in [20, 31]:
        for name, r in [("base", base), ("trained", trained)]:
            result[f"{name}_z{layer}"] = r[f"z{layer}"]
            result[f"{name}_cos{layer}"] = r[f"z{layer}"] / r[f"norm{layer}"]
            result[f"{name}_norm{layer}"] = r[f"norm{layer}"]
        result[f"delta_z{layer}"] = trained[f"z{layer}"] - base[f"z{layer}"]
        result[f"delta_cos{layer}"] = (
            result[f"trained_cos{layer}"] - result[f"base_cos{layer}"]
        )
    result["base_score"] = base["score"]
    result["trained_score"] = trained["score"]
    result["score_delta"] = trained["score"] - base["score"]
    return result


def distribution(values: list[float]) -> dict:
    a = np.array(values, dtype=np.float64)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("invalid summary distribution")
    return {
        "mean": float(a.mean()),
        "mean_absolute": float(abs(a).mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
        "p10_p90": np.quantile(a, [0.1, 0.9]).tolist(),
        "fraction_positive": float((a > 0).mean()),
    }


def correlation(x: list[float], y: list[float]) -> float | None:
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def describe(rows: list[dict]) -> dict:
    fields = [
        f"{model}_{metric}{layer}"
        for layer in [20, 31]
        for model in ["base", "trained"]
        for metric in ["z", "cos", "norm"]
    ]
    fields += ["delta_z20", "delta_cos20", "delta_z31", "delta_cos31", "score_delta"]
    return {
        "rows": len(rows),
        **{f: distribution([r[f] for r in rows]) for f in fields},
        "correlations": {
            field: {
                "length": correlation(
                    [r[field] for r in rows], [r["prompt_tokens"] for r in rows]
                ),
                "teacher_target": correlation(
                    [r[field] for r in rows], [r["soft_target"] for r in rows]
                ),
                "trained_score": correlation(
                    [r[field] for r in rows], [r["trained_score"] for r in rows]
                ),
            }
            for field in ["base_z20", "trained_z20", "delta_z20", "delta_cos20"]
        },
        "hard_label_auroc": {
            field: float(
                roc_auc_score([r["label"] for r in rows], [r[field] for r in rows])
            )
            if len({r["label"] for r in rows}) == 2
            else None
            for field in ["base_z20", "trained_z20", "delta_z20"]
        },
    }
