"""Post-hoc OOD read of the ID-selected seed ensembles (``ensemble.py``).

Each cell's ensemble pick is the candidate the frozen rule already selected,
so this reads OOD only for probes whose OOD was read before; every other
column of a scored bank is dropped unread. Per-row logits of the picked seeds
are written next to the metrics, and each seed's metrics must reproduce the
frozen OOD read.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from experiments.probe_generalization.evaluate_ood import ROOT, report
from gleipnir.probes.store import Store
from gleipnir.probes.training import make_sweep, score

# Same probes and activations; allows GPU/library rounding in near-tied rows.
TOLERANCE = 2e-3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--population", default="full")
    parser.add_argument("--backbone", choices=("prompted", "ntp"), required=True)
    parser.add_argument("--name", default="full01")
    args = parser.parse_args()
    results = ROOT / "results/probe_generalization" / args.name
    run = results / f"{args.backbone}-fp8"
    config = json.loads((run / "config.json").read_text())
    picks = json.loads((results / "ensemble_id.json").read_text())[args.backbone][
        "selection_ensemble"
    ]
    frozen = json.loads((results / f"ood_{args.backbone}.json").read_text())["pooled"]

    skip = 1 if args.backbone == "ntp" else 0
    store = Store(
        Path(args.store_root) / args.population / args.backbone / "ood",
        "fp8",
        skip=skip,
    )
    output = {"backbone": args.backbone, "ids": [r["id"] for r in store.rows]}
    for objective, cell in sorted(picks.items()):
        pick = cell["selected"]
        names = [
            f"{pick['architecture']}-{objective}-s{seed}" for seed in pick["seeds"]
        ]
        reference = frozen[objective]["selected"]
        if (reference["architecture"], reference["epoch"]) != (
            pick["architecture"],
            pick["epoch"],
        ):
            raise ValueError(f"{objective}: ensemble pick is not the frozen pick")
        sweep = make_sweep(
            config["architectures"],
            config["objectives"],
            config["seeds"],
            store.meta["hidden"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            device="cuda",
        )
        state = torch.load(run / f"probes_e{pick['epoch']}.pt", map_location="cuda")
        if [tuple(s) for s in state["specs"]] != [
            (s.name, s.architecture) for s in sweep.specs
        ]:
            raise ValueError(f"probe order differs in {run}")
        sweep.bank.load_state_dict(state["bank"])
        scored = score(
            sweep, store, config["batch"], "cuda", config.get("token_budget")
        )
        logits = {n: scored[n] for n in names}  # every other column is dropped
        del scored

        per_seed = {n: report(store.rows, v) for n, v in logits.items()}
        drift = max(
            abs(
                r["macro"]["pauroc_at_20"]
                - reference["ood_per_probe"][n]["macro"]["pauroc_at_20"]
            )
            for n, r in per_seed.items()
        )
        if drift > TOLERANCE:
            raise ValueError(
                f"{objective}: per-seed OOD differs from frozen by {drift}"
            )
        mean = np.mean([np.asarray(v, dtype=float) for v in logits.values()], 0)
        output[objective] = {
            "architecture": pick["architecture"],
            "epoch": pick["epoch"],
            "id_ensemble": pick["ensemble"],
            "id_seed_median": pick["seed_median"],
            "ood_seed_median": {
                k: statistics.median(r["macro"][k] for r in per_seed.values())
                for k in ("pauroc_at_20", "auroc", "brier")
            },
            "ood_ensemble": report(store.rows, mean),
            "ood_per_seed": per_seed,
            "max_seed_drift_from_frozen": drift,
            "logits": logits,
        }
        e = output[objective]
        print(
            f"{args.backbone} {objective}: {pick['architecture']} e{pick['epoch']} "
            f"OOD seed median {e['ood_seed_median']['pauroc_at_20']:.4f} "
            f"ensemble {e['ood_ensemble']['macro']['pauroc_at_20']:.4f}",
            flush=True,
        )
    (results / f"ensemble_ood_{args.backbone}.json").write_text(
        json.dumps(output) + "\n"
    )


if __name__ == "__main__":
    main()
