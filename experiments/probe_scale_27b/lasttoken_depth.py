"""Descriptive OOD depth curve for the 27B last-token probes (frozen 2026-10-28).

For each fixed layer and objective, the C chosen on ID by the frozen rule
(pAUROC@20, then AUROC, then lower Brier) from the run's existing
``lasttoken_<backbone>.json`` sweep is refit on train and scored on OOD once.
Nothing here selects on OOD; the run's selected last-token probe is unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.probe_generalization.lasttoken import fit, load, macro

ROOT = Path(__file__).resolve().parents[2]
LAYERS = (24, 28, 32, 36, 40, 44)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", required=True, help="<root>/full/<backbone>")
    parser.add_argument("--backbone", choices=("prompted", "ntp"), required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    results = ROOT / "results/probe_generalization" / args.name
    sweep = json.loads((results / f"lasttoken_{args.backbone}.json").read_text())
    root = Path(args.store_root) / "full" / args.backbone
    train_x, train_rows = load(root / "train")
    ood_x, ood_rows = load(root / "ood")
    out = {"backbone": args.backbone, "layers": list(LAYERS), "probes": []}
    for layer in LAYERS:
        for objective in ("hard", "soft"):
            best = max(
                (
                    p
                    for p in sweep["probes"]
                    if p["layer"] == layer and p["objective"] == objective
                ),
                key=lambda p: (
                    p["id"]["pauroc_at_20"],
                    p["id"]["auroc"],
                    -p["id"]["brier"],
                ),
            )
            model = fit(train_x[:, layer], train_rows, objective, best["C"])
            scores = model.predict_proba(ood_x[:, layer])[:, 1]
            record = {**best, "ood": macro(ood_rows, scores)}
            out["probes"].append(record)
            print(
                f"{args.backbone} layer {layer} {objective} C={best['C']} "
                f"ID {best['id']['pauroc_at_20']:.4f} "
                f"OOD {record['ood']['pauroc_at_20']:.4f}",
                flush=True,
            )
    (results / f"lasttoken_depth_{args.backbone}.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
