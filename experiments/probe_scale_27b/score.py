"""Apply the frozen ID selection rule to one pilot sweep, then read OOD once.

Selection is the probe study's rule (``evaluate_ood.select``): per objective,
the (architecture, epoch) candidate with the best seed-median ID pAUROC@20,
then AUROC, then lower Brier, plus each family's ID-best candidate. Only those
probes score the OOD sample, and their per-row logits are kept for the paired
comparison with the LoRA monitors. Pilot sweeps end at epoch 16, so the
budget-extension rule never triggers.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from experiments.probe_generalization.evaluate_ood import report, select
from gleipnir.probes.store import Store
from gleipnir.probes.training import make_sweep, score

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="sweep dir with summary.json")
    parser.add_argument("--ood-store", required=True, help="NTP OOD store dir")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    if config["backbone"] != "ntp":
        raise ValueError("the scale pilot uses the NTP backbone only")
    chosen = select(summary, "fp8")
    (out / "selection.json").write_text(json.dumps(chosen, indent=2) + "\n")

    picks = [c["selected"] for c in chosen.values()] + [
        f for c in chosen.values() for f in c["families"].values()
    ]
    wanted = defaultdict(set)
    for pick in picks:
        wanted[pick["epoch"]].update(pick["probes"])
    store = Store(Path(args.ood_store), "fp8", skip=1)  # drop the attention sink
    ids = [r["id"] for r in store.rows]
    kept = {}
    for epoch, names in sorted(wanted.items()):
        sweep = make_sweep(
            config["architectures"],
            config["objectives"],
            config["seeds"],
            store.meta["hidden"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            device="cuda",
        )
        state = torch.load(run / f"probes_e{epoch}.pt", map_location="cuda")
        if [tuple(s) for s in state["specs"]] != [
            (s.name, s.architecture) for s in sweep.specs
        ]:
            raise ValueError(f"probe order differs in {run}/probes_e{epoch}.pt")
        sweep.bank.load_state_dict(state["bank"])
        logits = score(
            sweep, store, config["batch"], "cuda", config.get("token_budget")
        )
        kept[epoch] = {n: logits[n] for n in names}  # every other column is dropped
        print(f"epoch {epoch}: scored {len(names)} selected probes", flush=True)
    (out / "ood_logits.json").write_text(
        json.dumps({"ids": ids, "logits": {str(e): v for e, v in kept.items()}}) + "\n"
    )

    def evaluate(pick: dict) -> dict:
        per_probe = {
            n: report(store.rows, kept[pick["epoch"]][n]) for n in pick["probes"]
        }
        return {
            **pick,
            "ood_seed_median_pauroc_at_20": statistics.median(
                r["macro"]["pauroc_at_20"] for r in per_probe.values()
            ),
            "ood_per_probe": per_probe,
        }

    result = {
        "run": str(run),
        "ood_store": args.ood_store,
        "ood_rows": len(ids),
        "selection_rule": "ID seed-median pAUROC@20, then AUROC, then lower Brier",
        "pooled": {
            objective: {
                "selected": evaluate(c["selected"]),
                "families": {k: evaluate(v) for k, v in c["families"].items()},
            }
            for objective, c in chosen.items()
        },
    }
    (out / "ood.json").write_text(json.dumps(result, indent=2) + "\n")
    for objective, cell in result["pooled"].items():
        s = cell["selected"]
        print(
            f"{objective}: {s['architecture']} e{s['epoch']} "
            f"ID {s['id_seed_median']['pauroc_at_20']:.4f} "
            f"OOD {s['ood_seed_median_pauroc_at_20']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
