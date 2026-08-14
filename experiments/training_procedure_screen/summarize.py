#!/usr/bin/env python3
"""Summarize final, checkpoint, ensemble, and calibration diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapter_capacity_scaling.prepare import read_jsonl  # noqa: E402
from gleipnir.binary_evaluation import metric_views  # noqa: E402


def primary_metrics(result: dict[str, Any]) -> dict[str, float]:
    macro = result["metrics"]["macro"]["macro"]
    pooled = result["metrics"]["pooled"]
    return {
        "macro_auroc": float(macro["auroc"]),
        "macro_balanced_accuracy": float(macro["balanced_accuracy"]),
        "macro_brier": float(macro["brier"]),
        "pooled_auroc": float(pooled["auroc"]),
        "pooled_balanced_accuracy": float(pooled["balanced_accuracy"]),
        "pooled_brier": float(pooled["brier"]),
    }


def result_row(job: dict[str, Any], checkpoint: str) -> dict[str, Any]:
    path = Path(job["output_dir"]) / "causal_validation" / checkpoint / "result.json"
    result = json.loads(path.read_text())
    return {
        "job_name": job["job_name"],
        "checkpoint": checkpoint,
        "intervention_family": job["intervention_family"],
        "seed": int(job["seed"]),
        "evaluation_base": job.get("evaluation_base", "bf16"),
        "promotion_eligible_by_design": bool(job.get("promotion_eligible", True)),
        "train_rows": int(job["train_rows"]),
        "effective_batch_size": int(job["effective_batch_size"]),
        "quantization_enabled": bool(job["quantization_enabled"]),
        "lora_init": job["lora_init"],
        "lora_use_dora": bool(job["lora_use_dora"]),
        "lora_target_modules": ",".join(job["lora_target_modules"]),
        "soft_loss_type": job["soft_loss_type"],
        "soft_huber_delta": float(job["soft_huber_delta"]),
        "direct_loss_weight": float(job["direct_loss_weight"]),
        "dataset_loss_weighting": job["dataset_loss_weighting"],
        **primary_metrics(result),
    }


def read_predictions(job: dict[str, Any], checkpoint: str) -> pd.DataFrame:
    path = (
        Path(job["output_dir"])
        / "causal_validation"
        / checkpoint
        / "predictions.jsonl"
    )
    frame = pd.read_json(path, lines=True)
    return frame.sort_values(["dataset", "index"]).reset_index(drop=True)


def logit_average(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Average aligned prediction logits and return a prediction frame."""
    if not frames:
        raise ValueError("an ensemble requires at least one prediction frame")
    reference = frames[0][["dataset", "index", "label"]].reset_index(drop=True)
    logits = []
    for frame in frames:
        keys = frame[["dataset", "index", "label"]].reset_index(drop=True)
        if not keys.equals(reference):
            raise ValueError("ensemble prediction identities are not aligned")
        scores = frame["score"].to_numpy(dtype=float)
        scores = np.clip(scores, 1e-12, 1.0 - 1e-12)
        logits.append(np.log(scores / (1.0 - scores)))
    mean_logit = np.mean(np.stack(logits), axis=0)
    return reference.assign(score=1.0 / (1.0 + np.exp(-mean_logit)))


def macro_ba(frame: pd.DataFrame, threshold: float) -> float:
    values = []
    for _, group in frame.groupby("dataset", sort=True):
        labels = group["label"].to_numpy(dtype=int)
        predictions = group["score"].to_numpy(dtype=float) >= threshold
        positives = labels == 1
        negatives = labels == 0
        if positives.any() and negatives.any():
            values.append(
                0.5
                * (
                    float(predictions[positives].mean())
                    + float((~predictions[negatives]).mean())
                )
            )
    return float(np.mean(values))


def best_macro_threshold(frame: pd.DataFrame) -> float:
    """Select a deterministic global threshold from bounded score quantiles."""
    scores = frame["score"].to_numpy(dtype=float)
    candidates = np.unique(
        np.concatenate((np.quantile(scores, np.linspace(0, 1, 257)), [0.5]))
    )
    ranked = sorted(
        (
            (macro_ba(frame, float(value)), -abs(float(value) - 0.5), float(value))
            for value in candidates
        ),
        reverse=True,
    )
    return ranked[0][2]


def deterministic_folds(frame: pd.DataFrame, folds: int = 5) -> np.ndarray:
    return np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"{row.dataset}\0{row.index}".encode()).digest()[:8],
                "big",
            )
            % folds
            for row in frame.itertuples(index=False)
        ],
        dtype=int,
    )


def crossfit_calibration(frame: pd.DataFrame, folds: int = 5) -> dict[str, Any]:
    """Cross-fit a global threshold and Platt map without scoring fit rows."""
    fold_ids = deterministic_folds(frame, folds)
    threshold_predictions = np.zeros(len(frame), dtype=float)
    platt_scores = np.zeros(len(frame), dtype=float)
    thresholds = []
    raw = np.clip(frame["score"].to_numpy(dtype=float), 1e-8, 1.0 - 1e-8)
    features = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    labels = frame["label"].to_numpy(dtype=int)
    for fold in range(folds):
        held = fold_ids == fold
        fitting = ~held
        fit_frame = frame.loc[fitting].reset_index(drop=True)
        threshold = best_macro_threshold(fit_frame)
        thresholds.append(threshold)
        threshold_predictions[held] = (raw[held] >= threshold).astype(float)
        calibrator = LogisticRegression(C=1e6, solver="lbfgs")
        calibrator.fit(features[fitting], labels[fitting])
        platt_scores[held] = calibrator.predict_proba(features[held])[:, 1]
    threshold_frame = frame.assign(score=threshold_predictions)
    platt_frame = frame.assign(score=platt_scores)
    return {
        "folds": folds,
        "thresholds": thresholds,
        "threshold_macro_balanced_accuracy": metric_views(threshold_frame)["macro"][
            "macro"
        ]["balanced_accuracy"],
        "platt_macro_balanced_accuracy": metric_views(platt_frame)["macro"]["macro"][
            "balanced_accuracy"
        ],
        "platt_macro_brier": metric_views(platt_frame)["macro"]["macro"]["brier"],
    }


def write_ensemble(
    name: str, frame: pd.DataFrame, members: list[str], output_dir: Path
) -> dict[str, Any]:
    destination = output_dir / "ensembles" / name
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_json(
        destination / "predictions.jsonl",
        orient="records",
        lines=True,
        double_precision=15,
    )
    result = {"name": name, "members": members, "metrics": metric_views(frame)}
    (destination / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("results/training_procedure_screen/lambda_jobs.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/training_procedure_screen/summary"),
    )
    args = parser.parse_args()
    jobs = read_jsonl(args.jobs.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    final = pd.DataFrame([result_row(job, "final") for job in jobs])
    checkpoint_rows = []
    for job in jobs:
        validation_dir = Path(job["output_dir"]) / "causal_validation"
        for path in validation_dir.glob("checkpoint-*/result.json"):
            checkpoint_rows.append(result_row(job, path.parent.name))
    checkpoints = pd.DataFrame(checkpoint_rows)
    final.to_csv(output_dir / "results.csv", index=False)
    checkpoints.to_csv(output_dir / "checkpoint_results.csv", index=False)

    baseline = final[final["job_name"] == "baseline-seed0"].iloc[0]
    contrasts = final.copy()
    for metric in ("macro_auroc", "macro_balanced_accuracy", "macro_brier"):
        contrasts[f"difference_from_baseline_{metric}"] = (
            contrasts[metric] - float(baseline[metric])
        )
    contrasts["eligible"] = (
        contrasts["promotion_eligible_by_design"]
        & (contrasts["difference_from_baseline_macro_auroc"] > 0)
        & (contrasts["difference_from_baseline_macro_balanced_accuracy"] >= -0.002)
        & (contrasts["difference_from_baseline_macro_brier"] <= 0.002)
    )
    contrasts.to_csv(output_dir / "contrasts.csv", index=False)

    baseline_jobs = [
        next(job for job in jobs if job["job_name"] == f"baseline-seed{seed}")
        for seed in (0, 1, 2)
    ]
    seed_frames = [read_predictions(job, "final") for job in baseline_jobs]
    seed_ensemble_frame = logit_average(seed_frames)
    seed_ensemble = write_ensemble(
        "baseline-seeds-0-1-2-logit-mean",
        seed_ensemble_frame,
        [job["job_name"] for job in baseline_jobs],
        output_dir,
    )
    baseline0 = baseline_jobs[0]
    baseline0_checkpoints = sorted(
        checkpoints[checkpoints["job_name"] == "baseline-seed0"]["checkpoint"],
        key=lambda value: int(str(value).removeprefix("checkpoint-")),
    )
    checkpoint_frames = [
        read_predictions(baseline0, checkpoint) for checkpoint in baseline0_checkpoints
    ]
    checkpoint_ensembles = []
    if checkpoint_frames:
        checkpoint_ensembles.append(
            write_ensemble(
                "baseline-seed0-all-checkpoints-logit-mean",
                logit_average(checkpoint_frames),
                baseline0_checkpoints,
                output_dir,
            )
        )
        checkpoint_ensembles.append(
            write_ensemble(
                "baseline-seed0-last-two-checkpoints-logit-mean",
                logit_average(checkpoint_frames[-2:]),
                baseline0_checkpoints[-2:],
                output_dir,
            )
        )
    calibration = {
        job["job_name"]: crossfit_calibration(read_predictions(job, "final"))
        for job in baseline_jobs
    }
    eligible = contrasts[
        contrasts["eligible"] & (contrasts["intervention_family"] != "baseline")
    ].sort_values("macro_auroc", ascending=False)
    summary = {
        "primary_metric": "macro_auroc",
        "baseline_seed0": baseline.to_dict(),
        "baseline_seed_mean": {
            metric: float(
                final[final["intervention_family"] == "baseline"][metric].mean()
            )
            for metric in ("macro_auroc", "macro_balanced_accuracy", "macro_brier")
        },
        "eligible_single_seed_cells": eligible["job_name"].tolist(),
        "promotion_permitted": False,
        "promotion_requirement": "add seeds 1 and 2 to an eligible intervention",
        "seed_ensemble": seed_ensemble,
        "checkpoint_ensembles": checkpoint_ensembles,
        "crossfit_baseline_calibration": calibration,
        "competition_validation_used": False,
        "final_test_used": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
