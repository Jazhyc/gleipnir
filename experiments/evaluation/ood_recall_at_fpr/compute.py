"""Recall at a fixed false-positive rate on the frozen OOD suite.

For every monitor in ``config.json``, load its per-row OOD scores, compute the
recall at the configured false-positive rate for each source and average the
sources with equal weight. Source-macro pAUROC@20 is recomputed from the same
rows and must match the frontier registry, which checks the files and joins.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from gleipnir.adversarial.metrics import recall_at_fpr
from gleipnir.evaluation.metrics import normalized_partial_auroc
from gleipnir.repository import repo_root

EXPERIMENT = Path(__file__).resolve().parent
REPOSITORY_ROOT = repo_root(EXPERIMENT)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results/ood_recall_at_fpr/recall_at_fpr.json"
LABEL_FIELDS = ("label", "ground_truth", "metadata.ground_truth")


def read_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def nested(row: dict[str, Any], dotted: str) -> Any:
    value: Any = row
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def load_reference(config: dict[str, Any]) -> dict[str, tuple[int, str]]:
    """Return ``{row id: (label, source)}`` for the frozen suite."""
    spec = config["reference"]
    reference = {
        row[spec["id"]]: (int(row[spec["label"]]), row[spec["source"]])
        for row in read_rows(REPOSITORY_ROOT / spec["file"])
    }
    if len(reference) != config["rows"]:
        raise ValueError(
            f"reference has {len(reference)} rows, expected {config['rows']}"
        )
    return reference


def load_scores(
    monitor: dict[str, Any], reference: dict[str, tuple[int, str]]
) -> dict[str, float]:
    """Return ``{row id: score}`` and check coverage and any labels the file carries."""
    scores: dict[str, float] = {}
    for file in monitor["files"]:
        for row in read_rows(REPOSITORY_ROOT / file):
            row_id = row["id"]
            if row_id in scores:
                raise ValueError(f"{monitor['name']}: duplicate id {row_id}")
            scores[row_id] = float(row[monitor["score_field"]])
            for field in LABEL_FIELDS:
                label = nested(row, field)
                if label is not None and int(label) != reference[row_id][0]:
                    raise ValueError(f"{monitor['name']}: label mismatch for {row_id}")
    # Rows whose input the provider rejected score a fixed uninformative value,
    # as in the campaign that produced them.
    for file in monitor.get("rejected_files", []):
        for row in read_rows(REPOSITORY_ROOT / file):
            if row["id"] in scores:
                raise ValueError(f"{monitor['name']}: rejected id {row['id']} scored")
            scores[row["id"]] = float(monitor["rejected_score"])
    if set(scores) != set(reference):
        raise ValueError(
            f"{monitor['name']}: scores do not cover the frozen suite exactly"
        )
    return scores


def evaluate(
    scores: dict[str, float], reference: dict[str, tuple[int, str]], fpr: float
) -> dict[str, Any]:
    """Per-source and source-macro recall at ``fpr``, with pAUROC@20 as a check."""
    per_source: dict[str, dict[str, Any]] = {}
    for source in sorted({source for _, source in reference.values()}):
        ids = [
            row_id
            for row_id, (_, row_source) in reference.items()
            if row_source == source
        ]
        labels = np.array([reference[row_id][0] for row_id in ids])
        values = np.array([scores[row_id] for row_id in ids])
        at_fpr = recall_at_fpr(labels, values, fpr)
        per_source[source] = {
            "recall": at_fpr["recall"],
            "achieved_fpr": at_fpr["fpr"],
            "threshold": at_fpr["threshold"],
            "benign": at_fpr["honest"],
            "harmful": at_fpr["harmful"],
            "pauroc20": normalized_partial_auroc(labels, values, max_fpr=0.2),
            "auroc": float(roc_auc_score(labels, values)),
            "unique_scores": int(len(np.unique(values))),
        }
    return {
        "mean_recall": float(
            np.mean([entry["recall"] for entry in per_source.values()])
        ),
        "mean_auroc": float(np.mean([entry["auroc"] for entry in per_source.values()])),
        "mean_pauroc20": float(
            np.mean([entry["pauroc20"] for entry in per_source.values()])
        ),
        "per_source": per_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT / "config.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    reference = load_reference(config)
    results = []
    for monitor in config["monitors"]:
        metrics = evaluate(load_scores(monitor, reference), reference, config["fpr"])
        gap = abs(metrics["mean_pauroc20"] - monitor["registry_pauroc20"])
        if gap > config["pauroc_tolerance"]:
            raise ValueError(
                f"{monitor['name']}: pAUROC@20 {metrics['mean_pauroc20']:.4f} does not "
                f"reproduce the registry value {monitor['registry_pauroc20']:.4f}"
            )
        print(
            f"{monitor['name']:32s} recall {metrics['mean_recall']:.4f}  "
            f"AUROC {metrics['mean_auroc']:.4f}  "
            f"pAUROC@20 {metrics['mean_pauroc20']:.4f}"
        )
        results.append(
            {
                "monitor": monitor["name"],
                "group": monitor["group"],
                "cost_usd_per_1k": monitor["cost_usd_per_1k"],
                "registry_pauroc20": monitor["registry_pauroc20"],
                **metrics,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"fpr": config["fpr"], "monitors": results}, indent=1) + "\n"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
