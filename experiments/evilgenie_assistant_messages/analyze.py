"""Scores with and without assistant messages, paired on the same runs.

``PYTHONPATH=src:. python -m experiments.evilgenie_assistant_messages.analyze``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.evilgenie_api_baselines import run as api
from experiments.evilgenie_released_stack.analyze import ood_line
from gleipnir.evaluation.metrics import normalized_partial_auroc

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(Path(__file__).with_name("config.json").read_text())
OUT = ROOT / CONFIG["results_dir"]


def scores(path: Path) -> dict[str, float]:
    return {r["id"]: float(r["score"]) for r in api.read_jsonl(path)}


def paired(
    y: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    groups: np.ndarray,
    draws: int = 10_000,
) -> dict:
    """Both metrics before and after, and their difference, with 95%
    bootstrap intervals that resample whole groups (problems or runs)."""
    rng = np.random.default_rng(0)
    clusters = [np.flatnonzero(groups == g) for g in sorted(set(groups))]
    out: dict = {}
    for name, fn in (
        ("auroc", roc_auc_score),
        ("pauroc_at_20", normalized_partial_auroc),
    ):
        point = (fn(y, before), fn(y, after))
        diffs = []
        for _ in range(draws):
            d = np.concatenate(
                [clusters[k] for k in rng.integers(0, len(clusters), len(clusters))]
            )
            if len(set(y[d])) == 2:
                diffs.append(fn(y[d], after[d]) - fn(y[d], before[d]))
        out[name] = {
            "without": float(point[0]),
            "with": float(point[1]),
            "difference": float(point[1] - point[0]),
            "difference_interval": np.percentile(diffs, [2.5, 97.5]).tolist(),
            "draws_used": len(diffs),
        }
    return out


def main() -> None:
    metas = {r["id"]: r["metadata"] for r in api.read_jsonl(ROOT / CONFIG["states"])}
    report: dict = {}
    for model in CONFIG["models"]:
        old = scores(ROOT / CONFIG["baseline_results"] / f"{model}.jsonl")
        new = scores(OUT / f"{model}.jsonl")
        if set(new) != set(metas) or set(old) != set(metas):
            raise ValueError(f"{model}: coverage differs")
        line = ood_line(model)
        entry: dict = {"ood_threshold": line}
        ids_all = [
            i for i, m in metas.items() if m["label_class"] in ("positive", "negative")
        ]
        subsets = {
            "all": (ids_all, "problem"),
            "with_assistant_text": (
                [i for i in ids_all if metas[i]["has_assistant_text"]],
                "problem",
            ),
            "arc191_c": (
                [i for i in ids_all if metas[i]["problem_id"] == api.MILD],
                "run",
            ),
        }
        for name, (ids, unit) in subsets.items():
            y = np.array(
                [metas[i]["label_class"] == "positive" for i in ids], dtype=int
            )
            b = np.array([old[i] for i in ids])
            a = np.array([new[i] for i in ids])
            g = np.array(
                [metas[i]["problem_id"] if unit == "problem" else i for i in ids]
            )
            entry[name] = {
                "hacks": int(y.sum()),
                "legitimate": int((1 - y).sum()),
                "resampled": unit,
                **paired(y, b, a, g),
                "flagged_hacks": [
                    int((b[y == 1] > line).sum()),
                    int((a[y == 1] > line).sum()),
                ],
                "flagged_legitimate": [
                    int((b[y == 0] > line).sum()),
                    int((a[y == 0] > line).sum()),
                ],
            }
        clean = [i for i, m in metas.items() if m["label_class"] == "clean_negative"]
        entry["flagged_clean_negative"] = [
            int(sum(old[i] > line for i in clean)),
            int(sum(new[i] > line for i in clean)),
            len(clean),
        ]
        report[model] = entry
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
