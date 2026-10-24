"""Apply the frozen ID selection rule, then score the frozen OOD suite once.

For each backbone and objective (a *cell*), candidates are (architecture,
checkpoint epoch) pairs over all seeds. A candidate's ID score is its
seed-median source-macro pAUROC@20, ties broken by seed-median AUROC, then
lower seed-median Brier. Selected per cell: the best candidate overall and the
best candidate of each architecture family (the AlphaEvolve penalty grid is one
family). Only those probes' OOD logits are kept; all other columns of a scored
bank are discarded unread. Last-token probes select (layer, C) per objective on
ID; the prompted backbone also reports the zero-shot decision margin.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gleipnir.evaluation.metrics import evaluate_binary_monitor
from gleipnir.probes.store import Store
from gleipnir.probes.training import make_sweep, score

ROOT = Path(__file__).resolve().parents[2]


def family(architecture: str) -> str:
    return "alphaevolve" if architecture.startswith("alphaevolve") else architecture


def report(rows: list[dict], logits) -> dict:
    scores = 1 / (1 + np.exp(-np.clip(np.asarray(logits, dtype=float), -60, 60)))
    frame = pd.DataFrame(
        {
            "dataset": [r["source"] for r in rows],
            "label": [r["label"] for r in rows],
            "score": scores,
        }
    )
    result = evaluate_binary_monitor(frame)
    return {"macro": result["macro"], "groups": result["groups"]}


def id_key(entries: list[dict], fmt: str) -> tuple[float, float, float]:
    macro = [e[fmt]["macro"] for e in entries]
    return (
        statistics.median(m["pauroc_at_20"] for m in macro),
        statistics.median(m["auroc"] for m in macro),
        -statistics.median(m["brier"] for m in macro),
    )


def describe(key: tuple, entries: list[dict], fmt: str) -> dict:
    return {
        "architecture": key[1],
        "epoch": key[2],
        "probes": sorted(e["name"] for e in entries),
        "id_seed_median": dict(
            zip(
                ("pauroc_at_20", "auroc", "neg_brier"),
                id_key(entries, fmt),
                strict=True,
            )
        ),
    }


def select(summary: dict, fmt: str) -> dict:
    """``{objective: {"selected": cand, "families": {family: cand}}}``."""
    candidates = defaultdict(list)
    for epoch, checkpoint in summary["checkpoints"].items():
        for name, entry in checkpoint["probes"].items():
            key = (entry["objective"], entry["architecture"], int(epoch))
            candidates[key].append({**entry, "name": name})
    chosen = {}
    for objective in sorted({k[0] for k in candidates}):
        mine = {k: v for k, v in candidates.items() if k[0] == objective}
        ranked = sorted(mine, key=lambda k: id_key(mine[k], fmt), reverse=True)

        families = {}
        for k in ranked:
            families.setdefault(family(k[1]), describe(k, mine[k], fmt))
        chosen[objective] = {
            "selected": describe(ranked[0], mine[ranked[0]], fmt),
            "families": families,
        }
    return chosen


def score_pooled(args, backbone: str, chosen: dict, ood_rows) -> dict:
    run = ROOT / "results/probe_generalization" / args.name / f"{backbone}-fp8"
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    picks = [c["selected"] for c in chosen.values()] + [
        f for c in chosen.values() for f in c["families"].values()
    ]
    wanted = defaultdict(set)
    for pick in picks:
        wanted[pick["epoch"]].update(pick["probes"])
    skip = 1 if backbone == "ntp" else 0
    store = Store(
        Path(args.store_root) / args.population / backbone / "ood", "fp8", skip=skip
    )
    hidden = store.meta["hidden"]
    kept = {}
    for epoch, names in sorted(wanted.items()):
        sweep = make_sweep(
            config["architectures"],
            config["objectives"],
            config["seeds"],
            hidden,
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
        del logits
        print(
            f"{backbone} epoch {epoch}: scored {len(names)} selected probes", flush=True
        )
    if [r["id"] for r in store.rows] != [r["id"] for r in ood_rows]:
        raise ValueError("OOD store rows differ from the extraction index")

    def evaluate(pick):
        per_probe = {
            n: report(store.rows, kept[pick["epoch"]][n]) for n in pick["probes"]
        }
        values = [r["macro"]["pauroc_at_20"] for r in per_probe.values()]
        best_id = max(
            pick["probes"],
            key=lambda n: summary["checkpoints"][str(pick["epoch"])]["probes"][n][
                "fp8"
            ]["macro"]["pauroc_at_20"],
        )
        return {
            **pick,
            "ood_seed_median_pauroc_at_20": statistics.median(values),
            "ood_best_id_seed": {"probe": best_id, **per_probe[best_id]},
            "ood_per_probe": per_probe,
        }

    return {
        objective: {
            "selected": evaluate(c["selected"]),
            "families": {k: evaluate(v) for k, v in c["families"].items()},
        }
        for objective, c in chosen.items()
    }


def last_token(args, backbone: str) -> dict:
    from experiments.probe_generalization.lasttoken import fit, load, macro

    root = Path(args.store_root) / args.population / backbone
    results = ROOT / "results/probe_generalization" / args.name
    sweep = json.loads((results / f"lasttoken_{backbone}.json").read_text())
    train_x, train_rows = load(root / "train")
    ood_x, ood_rows = load(root / "ood")
    out = {}
    for objective in ("hard", "soft"):
        best = max(
            (p for p in sweep["probes"] if p["objective"] == objective),
            key=lambda p: (
                p["id"]["pauroc_at_20"],
                p["id"]["auroc"],
                -p["id"]["brier"],
            ),
        )
        model = fit(train_x[:, best["layer"]], train_rows, objective, best["C"])
        scores = model.predict_proba(ood_x[:, best["layer"]])[:, 1]
        out[objective] = {**best, "ood": macro(ood_rows, scores)}
    if backbone == "prompted":
        margins = np.array([r["margin"] for r in ood_rows])
        out["zero_shot_margin"] = {"ood": macro(ood_rows, 1 / (1 + np.exp(-margins)))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--population", default="full")
    parser.add_argument("--backbone", choices=("prompted", "ntp"), required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    results = ROOT / "results/probe_generalization" / args.name
    summary = json.loads((results / f"{args.backbone}-fp8/summary.json").read_text())
    chosen = select(summary, "fp8")
    (results / f"selection_{args.backbone}.json").write_text(
        json.dumps(chosen, indent=2) + "\n"
    )
    ood_index = (
        Path(args.store_root) / args.population / args.backbone / "ood/index.jsonl"
    )
    ood_rows = [json.loads(line) for line in ood_index.open()]
    output = {
        "backbone": args.backbone,
        "selection_rule": "ID seed-median pAUROC@20, then AUROC, then lower Brier",
        "pooled": score_pooled(args, args.backbone, chosen, ood_rows),
        "last_token": last_token(args, args.backbone),
    }
    (results / f"ood_{args.backbone}.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    for objective, cell in output["pooled"].items():
        s = cell["selected"]
        print(
            f"{args.backbone} {objective}: {s['architecture']} e{s['epoch']} "
            f"ID {s['id_seed_median']['pauroc_at_20']:.4f} "
            f"OOD {s['ood_seed_median_pauroc_at_20']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
