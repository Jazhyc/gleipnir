"""Bootstrap intervals for Gleipnir 27B teacher-prompt Mean-OOD pAUROC@20.

Post-hoc, from saved predictions; no inference and no selection. Two schemes,
both over the unweighted six-source mean:

- ``row``: resample rows within each source x label stratum (the project's
  usual source-label-stratified bootstrap).
- ``shade_task``: as ``row``, but SHADE-Arena resamples whole task pairs
  (joined from upstream by ``source_row_index``). SHADE-Arena has only five
  task pairs, so this is a sensitivity check, not a reliable interval.

Neither covers seed variation (one seed) or serving-precision variation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from gleipnir.evaluation.metrics import normalized_partial_auroc

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/optimized_gap"
INPUTS = ROOT / "data/student_injection_awareness/regular/ood.jsonl"
SHADE_PARQUET = ROOT / (
    "data/tool_trajectory_monitoring/source/"
    "deliberative-monitor-pipeline-2f10f4e/filtered_eval/ctrl_shade.parquet"
)
RUNS = {
    "bf16": OUT / "bf16_teacher_ood.jsonl",
    "optimized": OUT / "opt27b_teacher_eu01/ood.jsonl",
}
RESAMPLES = 2000
SEED = 0


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def task_pair(trajectory_data: str) -> str:
    """As `shade_length_audit.analyze.task_pair`, without its campaign imports."""
    metadata = json.loads(trajectory_data)["metadata"]
    if isinstance(metadata, dict):
        return metadata["task_pair"]
    return re.search(r"'task_pair': '([^']+)'", metadata).group(1)


def interval(values: np.ndarray) -> dict:
    return {
        "se": float(values.std(ddof=1)),
        "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
    }


def stratified(rng, labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    out = []
    for _ in range(RESAMPLES):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        out.append(normalized_partial_auroc(labels[idx], scores[idx]))
    return np.array(out)


def clustered(rng, labels: np.ndarray, scores: np.ndarray, tasks: np.ndarray):
    members = [np.flatnonzero(tasks == t) for t in sorted(set(tasks))]
    out = []
    while len(out) < RESAMPLES:
        draw = rng.integers(len(members), size=len(members))
        idx = np.concatenate([members[k] for k in draw])
        if len(set(labels[idx])) == 2:  # skip single-class draws
            out.append(normalized_partial_auroc(labels[idx], scores[idx]))
    return np.array(out)


def main() -> None:
    inputs = read_jsonl(INPUTS)
    source = np.array([r["metadata"]["source_dataset"] for r in inputs])
    labels = np.array([int(r["metadata"]["ground_truth"]) for r in inputs])
    row = np.array(
        [r["metadata"]["original_metadata"]["source_row_index"] for r in inputs]
    )
    shade = pd.read_parquet(SHADE_PARQUET)
    is_shade = source == "shade_arena"
    if (shade["ground_truth"].to_numpy()[row[is_shade]] != labels[is_shade]).any():
        raise ValueError("SHADE source_row_index does not align with the parquet")
    pairs = [task_pair(t) for t in shade["trajectory_data"]]
    tasks = np.array([pairs[i] for i in row[is_shade]])
    groups = sorted(set(source))

    report = {
        "resamples": RESAMPLES,
        "seed": SEED,
        "shade_task_pairs": {
            t: {
                "rows": int((tasks == t).sum()),
                "positives": int(labels[is_shade][tasks == t].sum()),
            }
            for t in sorted(set(tasks))
        },
        "runs": {},
    }
    for name, path in RUNS.items():
        predictions = read_jsonl(path)
        if [p["id"] for p in predictions] != [r["id"] for r in inputs]:
            raise ValueError(f"{name}: prediction order/identity changed")
        scores = np.array([p["score"] for p in predictions])
        rng = np.random.default_rng(SEED)
        point, boots = {}, {}
        for g in groups:
            mask = source == g
            point[g] = normalized_partial_auroc(labels[mask], scores[mask])
            boots[g] = stratified(rng, labels[mask], scores[mask])
        task_boot = clustered(rng, labels[is_shade], scores[is_shade], tasks)
        row_macro = np.mean([boots[g] for g in groups], axis=0)
        task_macro = np.mean(
            [task_boot if g == "shade_arena" else boots[g] for g in groups], axis=0
        )
        report["runs"][name] = {
            "predictions": str(path.relative_to(ROOT)),
            "mean_ood": float(np.mean(list(point.values()))),
            "row": interval(row_macro),
            "shade_task": interval(task_macro),
            "shade_arena_task": interval(task_boot),
            "by_source": {
                g: {"pauroc_at_20": point[g], **interval(boots[g])} for g in groups
            },
        }
    path = OUT / "ood_bootstrap.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v["mean_ood"] for k, v in report["runs"].items()}), path)


if __name__ == "__main__":
    main()
