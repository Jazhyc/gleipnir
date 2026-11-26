"""Tabulate validation and OOD pAUROC@20 for the filter-margin arms."""

from __future__ import annotations

import json

from experiments.release_serving.targets import ROOT
from experiments.release_serving.validate import labels, macro_pauroc
from gleipnir.data.monitoring import read_rows, write_json

ARMS = ("control", "delta000", "delta005", "delta010", "delta020")
OUT = ROOT / "results/filter_margin_ood"
STRICT_OOD = ROOT / "results/teacher-disagreement-strict-ood01/evaluation/ood.jsonl"


def load(path) -> dict[str, float]:
    return {str(r["id"]): float(r["score"]) for r in read_rows(path)}


def main() -> None:
    report: dict = {}
    for arm in ARMS:
        entry: dict = {}
        for split in ("id", "ood"):
            path = OUT / arm / f"{split}.jsonl"
            if path.exists():
                entry[split] = macro_pauroc(load(path), labels(split), split)
        report[arm] = entry
    strict = OUT / "delta000/ood.jsonl"
    if strict.exists():
        new, old = load(strict), load(STRICT_OOD)
        ids = sorted(new)
        report["delta000_recorded_ood"] = {
            "recorded_pauroc_at_20": macro_pauroc(old, labels("ood"), "ood"),
            "score_mae": sum(abs(new[i] - old[i]) for i in ids) / len(ids),
        }
    write_json(OUT / "comparison.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
