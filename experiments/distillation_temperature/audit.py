"""Audit the Kimi K3 soft targets before choosing distillation temperatures."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from gleipnir.data.monitoring import file_hash, read_rows, write_json

ROOT = Path(__file__).resolve().parents[2]
TEACHER = "data/teacher-disagreement-delta000-sdpa04/teacher.jsonl"
TEACHER_SHA256 = "ba8b8b9cf5dc0a5d4e325be75612923d67cd250df31138c0c8b7c08e3d2e59bd"
STUDENT_ID = {
    "seed0": "results/teacher-seed-sdpa-s0-rescore01/evaluation/id.jsonl",
    "seed1": "results/teacher-seed-sdpa-s1-01/evaluation/id.jsonl",
    "seed2": "results/teacher-seed-sdpa-s2-01/evaluation/id.jsonl",
}
PROBABILITY_EDGES = [0, 1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.8, 0.95, 0.99, 0.999, 0.9999, 1]
TEMPERATURES = [0.5, 1.0, 2.0, 4.0, 8.0]
CONFIDENT_MARGIN = 4.0  # |logit| > 4 means p < 0.018 or p > 0.982.


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def kd_weights(margins: np.ndarray, temperature: float) -> np.ndarray:
    """Per-row gradient weight of T^2-scaled BCE for a small student error.

    d/dz_s [T^2 BCE(sigmoid(z_s/T), sigmoid(z_t/T))] ~= (z_s - z_t) * w with
    w = sigmoid'(z_t/T); T -> infinity tends to uniform logit matching.
    """
    p = sigmoid(margins / temperature)
    return p * (1 - p)


def margin_summary(margins: np.ndarray) -> dict:
    return {
        "rows": int(len(margins)),
        "quantiles": {
            str(q): float(v)
            for q, v in zip(
                (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1),
                np.quantile(margins, (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)),
                strict=True,
            )
        },
        "median_abs": float(np.median(np.abs(margins))),
        "share_abs_above_4": float((np.abs(margins) > CONFIDENT_MARGIN).mean()),
        "share_between_0.05_0.95": float(
            ((sigmoid(margins) > 0.05) & (sigmoid(margins) < 0.95)).mean()
        ),
    }


def audit(root: Path = ROOT) -> dict:
    if file_hash(root / TEACHER) != TEACHER_SHA256:
        raise ValueError("teacher cache checksum changed")
    rows = read_rows(root / TEACHER)
    p = np.array([float(r["soft_target"]) for r in rows])
    labels = np.array([int(r["label"]) for r in rows])
    logprobs = np.array(
        [
            [r["target_logprobs"]["negative"], r["target_logprobs"]["positive"]]
            for r in rows
        ]
    )
    margins = logprobs[:, 1] - logprobs[:, 0]
    if not np.allclose(sigmoid(margins), p, atol=1e-7) or not 0 < p.min() < p.max() < 1:
        raise ValueError("soft targets are not the binary-normalized logprobs")
    counts = Counter(np.round(margins, 6))
    grid = np.abs(margins * 8 - np.round(margins * 8))
    bins = []
    for lo, hi in zip(PROBABILITY_EDGES, PROBABILITY_EDGES[1:], strict=False):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        bins.append(
            {
                "range": [lo, hi],
                "rows": int(mask.sum()),
                "share": float(mask.mean()),
                "positive_labels": int(labels[mask].sum()),
            }
        )
    by_source = defaultdict(list)
    for row, margin in zip(rows, margins, strict=True):
        by_source[row["dataset"]].append(margin)
    confident = np.abs(margins) > CONFIDENT_MARGIN
    temperatures = {}
    for t in TEMPERATURES:
        w = kd_weights(margins, t)
        target = sigmoid(margins / t)
        temperatures[str(t)] = {
            "kd_weight_share_confident_rows": float(w[confident].sum() / w.sum()),
            "kd_weight_ratio_median_confident_to_uncertain": float(
                np.median(w[confident]) / np.median(w[~confident])
            ),
            "teacher_only_target_share_between_0.01_0.99": float(
                ((target > 0.01) & (target < 0.99)).mean()
            ),
        }
    students = {}
    for name, path in STUDENT_ID.items():
        values = read_rows(root / path)
        student = defaultdict(list)
        for value in values:
            student[value["source_dataset"]].append(float(value["margin"]))
        pooled = np.array([m for v in student.values() for m in v])
        students[name] = {
            "sha256": file_hash(root / path),
            "pooled": margin_summary(pooled),
            "by_source": {
                k: margin_summary(np.array(v)) for k, v in sorted(student.items())
            },
        }
    return {
        "teacher_cache": {"path": TEACHER, "sha256": TEACHER_SHA256},
        "rows": len(rows),
        "positive_labels": int(labels.sum()),
        "exact_zero_or_one": int(((p == 0) | (p == 1)).sum()),
        "margin_on_eighth_grid_share": float((grid < 1e-4).mean()),
        "unique_margins": len(counts),
        "rows_sharing_exact_margin_share": float(
            sum(c for c in counts.values() if c > 1) / len(rows)
        ),
        "label_teacher_side_disagreements": int(((p > 0.5) != (labels == 1)).sum()),
        "probability_bins": bins,
        "teacher_margins": margin_summary(margins),
        "teacher_margins_by_source": {
            k: margin_summary(np.array(v)) for k, v in sorted(by_source.items())
        },
        "temperatures": temperatures,
        "student_id_margins": students,
        "qualification": (
            "Student margins are on the ID evaluation set, teacher margins on the "
            "training set; populations differ, so scale comparisons are descriptive."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="target_audit01")
    args = parser.parse_args()
    out = ROOT / "results/distillation_temperature" / args.name / "audit.json"
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True)
    result = audit()
    write_json(out, result)
    print(out)


if __name__ == "__main__":
    main()
