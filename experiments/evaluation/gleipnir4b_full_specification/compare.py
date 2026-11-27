"""Compare Gleipnir 4B under the full specification with its compact rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.release_serving.targets import ROOT
from experiments.release_serving.validate import ID_SOURCES, labels
from gleipnir.data.monitoring import read_rows, write_json
from gleipnir.evaluation.metrics import evaluate_binary_monitor

OUT = ROOT / "results/gleipnir4b_full_specification"
# Recorded compact-instruction rows of the released 4B, per arm.
COMPACT = {
    "bf16": "results/qad-v02-bf16-01/evaluation",
    "optimized": "results/qad-v02-optimized01/evaluation",
}


def pauroc(scores: dict[str, float], truth: dict, split: str) -> dict:
    """Source-macro pAUROC@20 and its per-source values."""
    frame = pd.DataFrame(
        [
            {"source": truth[i][0], "label": truth[i][1], "score": s}
            for i, s in scores.items()
        ]
    )
    if split == "id":
        frame = frame.loc[frame["source"].isin(ID_SOURCES)]
    report = evaluate_binary_monitor(frame, group_column="source")
    return {
        "macro": report["macro"]["pauroc_at_20"],
        "sources": {g["group"]: g["pauroc_at_20"] for g in report["groups"]},
    }


def summarize(
    full: dict[str, float], compact: dict[str, float], truth: dict, split: str
) -> dict:
    if set(full) != set(compact):
        raise ValueError(f"{split}: full-specification and compact rows differ")
    ids = sorted(full)
    a = np.array([full[i] for i in ids])
    b = np.array([compact[i] for i in ids])
    new, old = pauroc(full, truth, split), pauroc(compact, truth, split)
    return {
        "rows": len(ids),
        "full_specification": new,
        "compact": old,
        "macro_difference": new["macro"] - old["macro"],
        "score_mean_abs_difference": float(np.abs(a - b).mean()),
        "threshold_flips": int(((a >= 0.5) != (b >= 0.5)).sum()),
    }


def load(path: Path) -> dict[str, float]:
    return {str(r["id"]): float(r["score"]) for r in read_rows(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=sorted(COMPACT))
    args = parser.parse_args()
    report: dict = {}
    for arm in args.arms or sorted(COMPACT):
        report[arm] = {}
        for split in ("id", "ood"):
            path = OUT / arm / f"{split}.jsonl"
            if not path.exists():
                continue
            compact = load(ROOT / COMPACT[arm] / f"{split}.jsonl")
            report[arm][split] = summarize(load(path), compact, labels(split), split)
    write_json(OUT / "comparison.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
