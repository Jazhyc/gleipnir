"""Compare the hard-label 27B with the soft 27B and the 4B hard/soft seeds.

Ranking and calibration use the seed-variation summary (source-macro
pAUROC@20, AUROC and Brier) on the same rows; saturation diagnostics count
distinct scores, tied rows and scores within 0.01 of 0 or 1. All arms were
served with BF16 vLLM 0.31.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from experiments.teachers.teacher_seed_variation.compare import (
    PREDICTIONS,
    load,
    summary,
)
from experiments.teachers.teacher_seed_variation.prepare import ROOT
from experiments.training.qwen38_27b_hard_labels.prepare import ARM, OOD_CONTEXT

HARD_4B = "results/teacher-hard-sdpa-s{seed}-01/evaluation/id.jsonl"
HARD_4B_OOD = "results/teacher-hard-sdpa-s{seed}-ood01/evaluation/ood.jsonl"
SOFT_4B = ("sdpa_seed0_rescored", "sdpa_seed1", "sdpa_seed2")
ARMS = {
    "id": {
        "27B hard, seed 0": f"results/{ARM}/evaluation/id.jsonl",
        "27B soft, seed 0": "results/qwen38-27b-s0-01/evaluation/id.jsonl",
        **{f"4B hard seed {s}": HARD_4B.format(seed=s) for s in range(3)},
        **{f"4B soft seed {s}": PREDICTIONS[k]["id"] for s, k in enumerate(SOFT_4B)},
    },
    "ood": {
        "27B hard, seed 0": f"results/{OOD_CONTEXT}/evaluation/ood.jsonl",
        "27B soft, seed 0": "results/qwen38-27b-s0-ood01/evaluation/ood.jsonl",
        **{f"4B hard seed {s}": HARD_4B_OOD.format(seed=s) for s in range(3)},
        **{f"4B soft seed {s}": PREDICTIONS[k]["ood"] for s, k in enumerate(SOFT_4B)},
    },
}


def saturation(rows: list[dict]) -> dict:
    scores = np.array([r["score"] for r in rows])
    counts = Counter(scores.tolist())
    return {
        "rows": len(rows),
        "unique_scores": len(counts),
        "tied_rows": sum(c for c in counts.values() if c > 1),
        "extreme_fraction": float(((scores <= 0.01) | (scores >= 0.99)).mean()),
    }


def main() -> None:
    report = {}
    for split, arms in ARMS.items():
        report[split] = {}
        for name, path in arms.items():
            if not (ROOT / path).is_file():
                continue
            rows = load(path, split)
            report[split][name] = {**summary(rows), "saturation": saturation(rows)}
    out = ROOT / "results/qwen38_27b_hard_labels/comparison.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    for split, arms in report.items():
        print(f"== {split}")
        for name, value in arms.items():
            m, s = value["macro"], value["saturation"]
            print(
                f"{name}: pAUC {m['pauroc_at_20']:.4f} AUROC {m['auroc']:.4f} "
                f"Brier {m['brier']:.4f} unique {s['unique_scores']} "
                f"tied {s['tied_rows']} extreme {s['extreme_fraction']:.3f} "
                + str({k: round(v, 4) for k, v in value["by_source"].items()})
            )


if __name__ == "__main__":
    main()
