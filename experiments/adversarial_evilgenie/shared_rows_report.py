"""Report EvilGenie Escalation numbers on the rows both Gleipnir monitors scored.

Post hoc, for the v1 technical report (Section 6.1, Appendix "Reward hacking by
other agent models"). Reads the frozen rows and both monitors' predictions and
prints AUROC and pAUROC@20 with 95% bootstrap intervals over problems (10,000
draws, seed 0), flag counts at each monitor's strict-OOD 1% line, and the
per-problem table.

    PYTHONPATH=src python experiments/adversarial_evilgenie/shared_rows_report.py
"""

import collections
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from gleipnir.evaluation.metrics import normalized_partial_auroc

# Data and results are git-ignored; point GLEIPNIR_DATA_ROOT at the checkout
# that holds them.
ROOT = Path(os.environ.get("GLEIPNIR_DATA_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data/adversarial_evilgenie/rows_4b-v03.jsonl"
OUT = ROOT / "results/adversarial_evilgenie"
MILD = "lcb_arc191_c"


def load(monitor: str) -> dict:
    path = OUT / monitor / "rows/predictions.jsonl"
    rows = (json.loads(line) for line in path.open())
    return {r["id"]: r for r in rows if r["status"] == "ok"}


def main() -> None:
    meta = {}
    for line in DATA.open():
        r = json.loads(line)
        meta[r["id"]] = r["metadata"]
    report = json.loads((OUT / "report.json").read_text())
    preds = {"4b": load("4b-v03"), "27b": load("27b-merged")}
    lines = {
        "4b": report["4b-v03"]["lines"]["1%"]["threshold"],
        "27b": report["27b-merged"]["lines"]["1%"]["threshold"],
    }
    ids = [
        i
        for i in preds["27b"]
        if i in preds["4b"] and meta[i]["label_class"] in ("positive", "negative")
    ]
    y = np.array([meta[i]["label_class"] == "positive" for i in ids], dtype=int)
    prob = np.array([meta[i]["problem_id"] for i in ids])
    print(
        f"shared rows: {y.sum()} reward hacks, {(1 - y).sum()} legitimate, "
        f"{len(set(prob))} problems"
    )

    rng = np.random.default_rng(0)
    clusters = [np.flatnonzero(prob == p) for p in sorted(set(prob))]
    draws = [
        np.concatenate(
            [clusters[k] for k in rng.integers(0, len(clusters), len(clusters))]
        )
        for _ in range(10_000)
    ]

    def interval(stat, score, sel):
        vals = []
        for d in draws:
            d = d[sel[d]]
            if len(set(y[d])) == 2:
                vals.append(stat(y[d], score[d]))
        return np.percentile(vals, [2.5, 97.5])

    mild = prob == MILD
    for m, p in preds.items():
        score = np.array([p[i]["score"] for i in ids])
        for name, sel in (("all", np.ones(len(y), bool)), (f"without {MILD}", ~mild)):
            idx = np.flatnonzero(sel)
            a = roc_auc_score(y[idx], score[idx])
            pa = normalized_partial_auroc(y[idx], score[idx])
            alo, ahi = interval(roc_auc_score, score, sel)
            plo, phi = interval(normalized_partial_auroc, score, sel)
            print(
                f"{m} {name}: AUROC {a:.3f} [{alo:.3f}, {ahi:.3f}]  "
                f"pAUROC@20 {pa:.3f} [{plo:.3f}, {phi:.3f}]"
            )
        ln = lines[m]
        hit = score > ln
        print(
            f"  1% line {ln:.4f}: hacks {hit[y == 1].sum()}/{(y == 1).sum()}"
            f" (other problems {hit[(y == 1) & ~mild].sum()}/"
            f"{((y == 1) & ~mild).sum()},"
            f" {MILD} {hit[(y == 1) & mild].sum()}/{((y == 1) & mild).sum()}),"
            f" legitimate {hit[y == 0].sum()}/{(y == 0).sum()}"
        )
        for problem in sorted(set(prob)):
            s = prob == problem
            ys = y[s]
            auc = (
                roc_auc_score(ys, score[s]) if 0 < ys.sum() < len(ys) else float("nan")
            )
            print(
                f"    {problem}: hacks {ys.sum()} legit {(1 - ys).sum()} "
                f"AUROC {auc:.3f}"
                f" flagged {hit[s][ys == 1].sum()}/{hit[s][ys == 0].sum()}"
            )
    models = collections.Counter(
        meta[i]["model"] for i, t in zip(ids, y, strict=True) if t
    )
    print("agent models with reward hacks:", len(models))


if __name__ == "__main__":
    main()
