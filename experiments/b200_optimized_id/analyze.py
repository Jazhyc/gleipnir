"""Paired ID drift, ranking and calibration with repeat diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gleipnir.binary_evaluation import metric_views
from gleipnir.calibration import binary_calibration
from gleipnir.inference_benchmark import paired_score_summary


def views(rows: list[dict], scores: list[float]) -> dict:
    frame = pd.DataFrame(
        {
            "dataset": [r["dataset"] for r in rows],
            "label": [r["label"] for r in rows],
            "score": scores,
        }
    )
    calibration = {}
    for source in ["pooled", *sorted(frame["dataset"].unique())]:
        subset = frame if source == "pooled" else frame[frame["dataset"] == source]
        calibration[source] = binary_calibration(subset["label"], subset["score"], 10)
    return {"ranking": metric_views(frame), "calibration": calibration}


def compare(rows: list[dict], baseline: list[dict], runs: list[list[dict]]) -> dict:
    """Pair exact ordered identities before comparing observed score distributions."""
    paired = paired_score_summary([baseline], runs)
    if [(r["id"], r["prompt_sha256"]) for r in rows] != [
        (r["id"], r["prompt_sha256"]) for r in baseline
    ]:
        raise ValueError("metric label/prompt identity changed")
    if any(
        row["label"] != old["label"] or row["dataset"] != old["dataset"]
        for row, old in zip(rows, baseline, strict=True)
    ):
        raise ValueError("metric source/label identity changed")
    median_scores = [
        statistics.median(run[i]["score"] for run in runs) for i in range(len(rows))
    ]
    old_scores = [r["score"] for r in baseline]
    old, candidate = views(rows, old_scores), views(rows, median_scores)
    keys = ("pauroc_at_20", "auroc", "brier", "balanced_accuracy", "recall", "fpr")
    old_groups = {g["group"]: g for g in old["ranking"]["macro"]["groups"]}
    new_groups = {g["group"]: g for g in candidate["ranking"]["macro"]["groups"]}

    def delta(a, b):
        return {
            k: b[k] - a[k] if a[k] is not None and b[k] is not None else None
            for k in keys
        }

    differences = {
        "macro": delta(
            old["ranking"]["macro"]["macro"], candidate["ranking"]["macro"]["macro"]
        ),
        "pooled": delta(old["ranking"]["pooled"], candidate["ranking"]["pooled"]),
        "per_source": {s: delta(old_groups[s], new_groups[s]) for s in old_groups},
    }
    populations = {}
    for source in ["pooled", *sorted(old_groups)]:
        positions = [
            i
            for i, r in enumerate(rows)
            if source == "pooled" or r["dataset"] == source
        ]
        a, b = np.array(old_scores)[positions], np.array(median_scores)[positions]
        difference = b - a
        populations[source] = {
            "n": len(positions),
            "signed_mean_score_difference": float(difference.mean()),
            "mean_absolute_score_difference": float(np.abs(difference).mean()),
            "score_error_percentiles": {
                str(q): float(np.percentile(np.abs(difference), q))
                for q in (50, 95, 99, 100)
            },
            "pearson": float(np.corrcoef(a, b)[0, 1]),
            "spearman": float(spearmanr(a, b).statistic),
            "threshold_flips": int(np.sum((a >= 0.5) != (b >= 0.5))),
            "negative_to_positive": int(np.sum((a < 0.5) & (b >= 0.5))),
            "positive_to_negative": int(np.sum((a >= 0.5) & (b < 0.5))),
        }
    margin_drift = {}
    old_margins = np.array([r["margin"] for r in baseline])
    new_margins = np.median([[r["margin"] for r in run] for run in runs], axis=0)
    for source in populations:
        mask = np.array([source == "pooled" or r["dataset"] == source for r in rows])
        difference = new_margins[mask] - old_margins[mask]
        margin_drift[source] = {
            "signed_mean": float(difference.mean()),
            "mean_absolute_difference": float(np.abs(difference).mean()),
            "absolute_error_percentiles": {
                str(q): float(np.percentile(np.abs(difference), q))
                for q in (50, 95, 99, 100)
            },
        }
    ranges = [
        max(r[i]["score"] for r in runs) - min(r[i]["score"] for r in runs)
        for i in range(len(rows))
    ]
    return {
        "population": (
            "fixed canonical CoT-removed ID; same-adapter whole-stack drift, "
            "no selection"
        ),
        "baseline": old,
        "candidate_repeat_median": candidate,
        "metric_deltas": differences,
        "paired": paired,
        "score_drift": populations,
        "margin_drift": margin_drift,
        "repeat_metrics": [views(rows, [r["score"] for r in run]) for run in runs],
        "repeat_variation": {
            "measured": len(runs) > 1,
            "passes": len(runs),
            "mean_score_range": statistics.mean(ranges),
            "max_score_range": max(ranges),
            "threshold_unstable_ids": [
                rows[i]["id"]
                for i in range(len(rows))
                if len({r[i]["score"] >= 0.5 for r in runs}) > 1
            ],
        },
    }


def analyze(directory: Path) -> dict:
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("ID evaluation incomplete")
    settings = json.loads((directory / "settings.json").read_text())
    if len(summary["passes"]) != settings["repeats"]:
        raise ValueError("completed pass count differs from frozen settings")
    rows = json.loads((directory / "workload.json").read_text())
    baseline = json.loads((directory / "reference.json").read_text())
    runs = [
        json.loads((directory / f"repeat{i}.json").read_text())
        for i in range(len(summary["passes"]))
    ]
    result = compare(rows, baseline, runs)
    result["analysis_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    result["input_artifact_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            directory / "workload.json",
            directory / "reference.json",
            *[directory / f"repeat{i}.json" for i in range(len(summary["passes"]))],
        ]
    }
    (directory / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (directory / "executed_sources/analyze.py").write_bytes(Path(__file__).read_bytes())
    print("ID metric deltas", result["metric_deltas"])
    print("ID score drift", result["score_drift"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    analyze(args.directory)


if __name__ == "__main__":
    main()
