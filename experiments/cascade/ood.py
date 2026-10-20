"""Score the six pre-registered cascade conditions on the frozen OOD suite.

Frozen before scoring (see README): escalate when the 4B logit margin is at
most one of three thresholds chosen on ID (15/30/50% of ID rows), merging by
the logit average or by the rank-preserving reorder. All six are reported;
nothing is selected on OOD here.
"""

from __future__ import annotations

import json

import numpy as np

from experiments.cascade.cascade import HOURLY, MERGES, ROOT, THROUGHPUT, logit
from experiments.teacher_seed_variation.compare import load, summary
from gleipnir.data.monitoring import read_rows, write_json

OUT = ROOT / "results/cascade"
THRESHOLDS = {"15%": 1.375, "30%": 2.000, "50%": 2.625}  # |4B logit margin|
MERGE_RULES = ("logit_average", "reorder")
INPUTS = {
    "4b": "results/qad-v02-optimized01/evaluation/ood.jsonl",
    "27b": "results/optimized_gap/opt27b_teacher_eu01/ood.jsonl",
    "4b_tokens": "data/teacher-disagreement-strict-ood01/workload.jsonl",
    "27b_tokens": "data/optimized_gap/teacher_ood_workload.jsonl",
}


def metrics(rows, scores) -> dict:
    s = summary([{**r, "score": float(v)} for r, v in zip(rows, scores, strict=True)])
    return {
        "pauroc_at_20": s["macro"]["pauroc_at_20"],
        "auroc": s["macro"]["auroc"],
        "brier": s["macro"]["brier"],
        "pooled_pauroc_at_20": s["pooled"]["pauroc_at_20"],
        "by_source": {k: v for k, v in s["by_source"].items() if v is not None},
    }


def main() -> None:
    rows = load(INPUTS["4b"], "ood")
    ids = [r["id"] for r in rows]
    big = {r["id"]: r["score"] for r in read_rows(ROOT / INPUTS["27b"])}
    tokens4 = {
        r["id"]: r["prompt_tokens"] for r in read_rows(ROOT / INPUTS["4b_tokens"])
    }
    tokens27 = {
        r["id"]: r["prompt_tokens"] for r in read_rows(ROOT / INPUTS["27b_tokens"])
    }
    if not set(ids) == set(big) == set(tokens4) == set(tokens27):
        raise ValueError("OOD coverage differs between inputs")
    p4 = np.array([r["score"] for r in rows])
    p27 = np.array([big[i] for i in ids])
    rate = {k: HOURLY / 3600 / v for k, v in THROUGHPUT.items()}
    cost4 = np.array([tokens4[i] for i in ids]) * rate["4b"]
    cost27 = np.array([tokens27[i] for i in ids]) * rate["27b"]
    margin = np.abs(logit(p4))
    report = {
        "inputs": INPUTS,
        "thresholds": THRESHOLDS,
        "references": {
            "4b_only": {"cost_per_1k": 1000 * float(cost4.mean()), **metrics(rows, p4)},
            "27b_only": {
                "cost_per_1k": 1000 * float(cost27.mean()),
                **metrics(rows, p27),
            },
        },
        "conditions": [],
    }
    for label, threshold in THRESHOLDS.items():
        mask = margin <= threshold
        cost = 1000 * float((cost4 + mask * cost27).mean())
        by_source = {}
        for r, m in zip(rows, mask, strict=True):
            by_source.setdefault(r["source_dataset"], []).append(bool(m))
        for merge in MERGE_RULES:
            report["conditions"].append(
                {
                    "id_fraction": label,
                    "margin_threshold": threshold,
                    "merge": merge,
                    "escalated_fraction": float(mask.mean()),
                    "escalated_by_source": {
                        k: float(np.mean(v)) for k, v in by_source.items()
                    },
                    "cost_per_1k": cost,
                    **metrics(rows, MERGES[merge](p4, p27, mask)),
                }
            )
    write_json(OUT / "ood_eval.json", report)
    print(json.dumps({"conditions": len(report["conditions"])}), flush=True)


if __name__ == "__main__":
    main()
