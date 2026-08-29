"""Pure contracts for the Qwen3.5 reasoning-mode source canary."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.tool_trajectory_monitoring.prompting import PromptTemplate
from gleipnir.binary_evaluation import metric_views

PREDICTION_RE = re.compile(r"(?im)\bPrediction\s*:\s*<?([01])>?")


def sha256_text(value: str) -> str:
    """Hash one rendered prompt or generated continuation."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def reasoning_prompt(
    template: PromptTemplate,
    trajectory: str,
    reasoning_instruction: str,
) -> str:
    """Render the reasoning task with the same binary semantics and envelope."""
    if not reasoning_instruction.strip():
        raise ValueError("reasoning instruction must not be empty")
    if not trajectory.strip():
        raise ValueError("trajectory must not be empty")
    separator = "" if trajectory.endswith("\n") else "\n"
    return (
        f"{reasoning_instruction.rstrip()}\n"
        f"{template.trajectory_open}\n{trajectory}{separator}"
        f"{template.trajectory_close}\n"
    )


def prefix_before_terminal_prediction(generation: str) -> tuple[str, int | None]:
    """Remove only the final generated 0/1 while preserving its exact boundary."""
    matches = list(PREDICTION_RE.finditer(generation))
    if not matches:
        return generation.rstrip() + "\nPrediction:", None
    match = matches[-1]
    return generation[: match.start(1)], int(match.group(1))


def binary_entropy(score: float) -> float:
    """Return binary entropy in bits, including well-defined endpoints."""
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must lie in [0, 1]")
    if score in {0.0, 1.0}:
        return 0.0
    return -score * math.log2(score) - (1.0 - score) * math.log2(1.0 - score)


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one complete condition with ranking and uncertainty views."""
    if not rows:
        raise ValueError("cannot summarize an empty prediction set")
    frame = pd.DataFrame(rows)
    required = {"id", "source", "label", "score", "generation_tokens"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction rows are missing fields: {sorted(missing)}")
    if frame["id"].duplicated().any():
        raise ValueError("prediction rows contain duplicate IDs")
    scores = frame["score"].astype(float)
    labels = frame["label"].astype(int)
    metric_frame = pd.DataFrame(
        {
            "dataset": frame["source"],
            "index": frame["id"],
            "label": labels,
            "score": scores,
        }
    )
    entropies = [binary_entropy(float(score)) for score in scores]
    correctness = (scores >= 0.5).astype(int) == labels
    parse_errors = (
        frame["parse_error"].astype(bool)
        if "parse_error" in frame
        else pd.Series(False, index=frame.index)
    )
    finish_reasons = (
        frame["finish_reason"].astype(str)
        if "finish_reason" in frame
        else pd.Series("", index=frame.index)
    )
    think_close = (
        frame["contains_think_close"].astype(bool)
        if "contains_think_close" in frame
        else pd.Series(False, index=frame.index)
    )
    return {
        "rows": len(frame),
        "metrics": metric_views(metric_frame),
        "uncertainty": {
            "mean_binary_entropy_bits": float(np.mean(entropies)),
            "mean_confidence": float(np.mean(np.maximum(scores, 1.0 - scores))),
            "mean_entropy_correct": (
                float(np.mean(np.asarray(entropies)[correctness]))
                if correctness.any()
                else None
            ),
            "mean_entropy_incorrect": (
                float(np.mean(np.asarray(entropies)[~correctness]))
                if (~correctness).any()
                else None
            ),
        },
        "generation": {
            "tokens_total": int(frame["generation_tokens"].sum()),
            "tokens_mean": float(frame["generation_tokens"].mean()),
            "tokens_max": int(frame["generation_tokens"].max()),
            "parse_errors": int(parse_errors.sum()),
            "truncated": int((finish_reasons == "length").sum()),
            "think_close_rows": int(think_close.sum()),
        },
    }


def paired_condition_comparison(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report paired score movement and pooled metric changes."""
    baseline = pd.DataFrame(baseline_rows).set_index("id")
    candidate = pd.DataFrame(candidate_rows).set_index("id")
    if not baseline.index.is_unique or not candidate.index.is_unique:
        raise ValueError("condition rows contain duplicate IDs")
    if set(baseline.index) != set(candidate.index):
        raise ValueError("condition row IDs differ")
    candidate = candidate.loc[baseline.index]
    for field in ("source", "label"):
        if not baseline[field].equals(candidate[field]):
            raise ValueError(f"condition {field} values differ")
    left = baseline["score"].to_numpy(dtype=float)
    right = candidate["score"].to_numpy(dtype=float)
    labels = baseline["label"].to_numpy(dtype=int)
    delta = right - left
    correlation = (
        float(np.corrcoef(left, right)[0, 1])
        if np.std(left) > 0 and np.std(right) > 0
        else None
    )
    return {
        "pooled_auroc_baseline": float(roc_auc_score(labels, left)),
        "pooled_auroc_candidate": float(roc_auc_score(labels, right)),
        "pooled_auroc_delta": float(
            roc_auc_score(labels, right) - roc_auc_score(labels, left)
        ),
        "mean_score_delta": float(np.mean(delta)),
        "mean_absolute_score_delta": float(np.mean(np.abs(delta))),
        "maximum_absolute_score_delta": float(np.max(np.abs(delta))),
        "score_correlation": correlation,
        "prediction_flips": int(np.sum((left >= 0.5) != (right >= 0.5))),
    }
