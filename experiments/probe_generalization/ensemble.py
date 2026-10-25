"""Seed ensembles of the probe sweep on ID, from the saved per-row logits.

For every (backbone, objective, architecture, checkpoint epoch) candidate, the
ensemble score is the mean logit of its seeds. It is compared with the
seed-median of the single probes, the quantity the frozen selection rule ranks.
The frozen rule is then applied to the ensembles (ID pAUROC@20, then AUROC,
then lower Brier), per cell and per architecture family. No OOD is read.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.probe_generalization.evaluate_ood import family, report

ROOT = Path(__file__).resolve().parents[2]
NAME = re.compile(r"^(?P<architecture>.+)-(?P<objective>hard|soft)-s(?P<seed>\d+)$")


def key(macro: dict) -> tuple[float, float, float]:
    return (macro["pauroc_at_20"], macro["auroc"], -macro["brier"])


def ensembles(run: Path, rows: list[dict]) -> list[dict]:
    by_id = {r["id"]: r for r in rows}
    out = []
    for path in sorted(run.glob("logits_e*_fp8.json")):
        epoch = int(re.search(r"_e(\d+)_", path.name).group(1))
        saved = json.loads(path.read_text())
        ordered = [by_id[i] for i in saved["ids"]]
        groups = defaultdict(dict)
        for name, logits in saved["logits"].items():
            m = NAME.match(name)
            groups[(m["objective"], m["architecture"])][int(m["seed"])] = logits
        for (objective, architecture), seeds in sorted(groups.items()):
            singles = [report(ordered, v)["macro"] for v in seeds.values()]
            mean = np.mean([np.asarray(v, dtype=float) for v in seeds.values()], 0)
            ensemble = report(ordered, mean)
            out.append(
                {
                    "objective": objective,
                    "architecture": architecture,
                    "epoch": epoch,
                    "seeds": sorted(seeds),
                    "seed_median": {
                        k: statistics.median(s[k] for s in singles)
                        for k in ("pauroc_at_20", "auroc", "brier")
                    },
                    "seed_best": max(s["pauroc_at_20"] for s in singles),
                    "ensemble": {
                        k: ensemble["macro"][k]
                        for k in ("pauroc_at_20", "auroc", "brier")
                    },
                }
            )
    return out


def select(candidates: list[dict], score: str) -> dict:
    chosen = {}
    for objective in sorted({c["objective"] for c in candidates}):
        mine = sorted(
            (c for c in candidates if c["objective"] == objective),
            key=lambda c: key(c[score]),
            reverse=True,
        )
        families = {}
        for c in mine:
            families.setdefault(family(c["architecture"]), c)
        chosen[objective] = {"selected": mine[0], "families": families}
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="full01")
    parser.add_argument("--population", default="full")
    args = parser.parse_args()
    results = ROOT / "results/probe_generalization" / args.name
    rows_path = ROOT / "data/probe_generalization" / args.population / "id.jsonl"
    rows = [json.loads(line) for line in rows_path.open()]
    output = {}
    for backbone in ("prompted", "ntp"):
        candidates = ensembles(results / f"{backbone}-fp8", rows)
        output[backbone] = {
            "candidates": candidates,
            "selection_seed_median": select(candidates, "seed_median"),
            "selection_ensemble": select(candidates, "ensemble"),
        }
        for objective in ("hard", "soft"):
            single = output[backbone]["selection_seed_median"][objective]["selected"]
            ens = output[backbone]["selection_ensemble"][objective]["selected"]
            print(
                f"{backbone} {objective}: seed-median pick {single['architecture']} "
                f"e{single['epoch']} {single['seed_median']['pauroc_at_20']:.4f} "
                f"(its ensemble {single['ensemble']['pauroc_at_20']:.4f}); "
                f"ensemble pick {ens['architecture']} e{ens['epoch']} "
                f"{ens['ensemble']['pauroc_at_20']:.4f}"
            )
    (results / "ensemble_id.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
