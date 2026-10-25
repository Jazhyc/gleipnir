"""Post-hoc seed ensembles of every architecture on ID and OOD, three epoch rules.

Inputs: the saved ID logits (``logits_e*_fp8.json``) and the post-hoc OOD
logits (``ensemble_ood.py``). An ensemble is the mean logit of the 5 seeds.
Epoch rules, all chosen on ID or fixed in advance, never on OOD:

- ``ensemble_id``: one epoch for all seeds, where the ensemble's ID score peaks;
- ``per_seed_id``: each seed at its own ID-best epoch, then ensembled;
- ``final``: the last checkpoint for every seed (as the Gleipnir 4B average).

Each ensemble is reported beside the seed median at the same epochs. The ID
key is the frozen one: pAUROC@20, then AUROC, then lower Brier.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.probe_generalization.evaluate_ood import ROOT, report

NAME = re.compile(r"^(?P<architecture>.+)-(?P<objective>hard|soft)-s(?P<seed>\d+)$")
METRICS = ("pauroc_at_20", "auroc", "brier")


def macro(rows: list[dict], logits) -> dict:
    return {k: report(rows, logits)["macro"][k] for k in METRICS}


def key(m: dict) -> tuple[float, float, float]:
    return (m["pauroc_at_20"], m["auroc"], -m["brier"])


def load_split(results: Path, backbone: str, population: str) -> dict:
    """``{split: (rows, {(objective, arch): {epoch: {seed: logits}}})}``."""
    data = ROOT / "data/probe_generalization" / population
    out = {}
    run = results / f"{backbone}-fp8"
    id_rows = {r["id"]: r for r in map(json.loads, (data / "id.jsonl").open())}
    table = defaultdict(lambda: defaultdict(dict))
    rows = None
    for path in sorted(run.glob("logits_e*_fp8.json")):
        epoch = int(re.search(r"_e(\d+)_", path.name).group(1))
        saved = json.loads(path.read_text())
        rows = rows or [id_rows[i] for i in saved["ids"]]
        for name, logits in saved["logits"].items():
            m = NAME.match(name)
            table[(m["objective"], m["architecture"])][epoch][int(m["seed"])] = (
                np.asarray(logits, dtype=float)
            )
    out["id"] = (rows, table)

    ood = np.load(results / "ensemble_ood" / f"logits_{backbone}.npz")
    ood_rows = {r["id"]: r for r in map(json.loads, (data / "ood.jsonl").open())}
    table = defaultdict(lambda: defaultdict(dict))
    for p, name in enumerate(ood["names"]):
        m = NAME.match(str(name))
        for k, epoch in enumerate(ood["epochs"]):
            table[(m["objective"], m["architecture"])][int(epoch)][int(m["seed"])] = (
                ood["logits"][k, :, p].astype(float)
            )
    out["ood"] = ([ood_rows[str(i)] for i in ood["ids"]], table)
    return out


def evaluate(splits: dict, cell: tuple, choice: dict[int, int]) -> dict:
    """Ensemble and seed median for ``choice`` = ``{seed: epoch}``."""
    out = {"epochs": {str(s): e for s, e in sorted(choice.items())}}
    for split, (rows, table) in splits.items():
        logits = [table[cell][e][s] for s, e in choice.items()]
        singles = [macro(rows, v) for v in logits]
        out[split] = {
            "ensemble": macro(rows, np.mean(logits, 0)),
            "seed_median": {
                k: statistics.median(m[k] for m in singles) for k in METRICS
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="full01")
    parser.add_argument("--population", default="full")
    args = parser.parse_args()
    results = ROOT / "results/probe_generalization" / args.name
    output = {}
    for backbone in ("prompted", "ntp"):
        splits = load_split(results, backbone, args.population)
        id_rows, id_table = splits["id"]
        cells = {}
        for cell in sorted(id_table):
            epochs = sorted(id_table[cell])
            seeds = sorted(id_table[cell][epochs[0]])
            ensemble_epoch = max(
                epochs,
                key=lambda e: key(
                    macro(id_rows, np.mean([id_table[cell][e][s] for s in seeds], 0))
                ),
            )
            per_seed = {
                s: max(epochs, key=lambda e: key(macro(id_rows, id_table[cell][e][s])))
                for s in seeds
            }
            cells["/".join(cell)] = {
                "ensemble_id": evaluate(
                    splits, cell, {s: ensemble_epoch for s in seeds}
                ),
                "per_seed_id": evaluate(splits, cell, per_seed),
                "final": evaluate(splits, cell, {s: epochs[-1] for s in seeds}),
            }
            print(backbone, *cell, flush=True)
        output[backbone] = cells
    path = results / "ensemble_ood" / "report.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
