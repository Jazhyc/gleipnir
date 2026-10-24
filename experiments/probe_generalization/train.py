"""Train the pooled-probe sweep for one backbone and storage format.

All architectures, objectives and seeds train together from one shared batch
stream (:mod:`gleipnir.probes.training`). At every checkpoint epoch, each
probe scores the evaluation split in every requested format; logits, metrics
and probe weights are written under ``results/probe_generalization/<name>/``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gleipnir.evaluation.metrics import evaluate_binary_monitor
from gleipnir.probes.store import Store
from gleipnir.probes.training import length_batches, make_runs, score, train_epoch

ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURES = (
    "mean_linear",
    "mean_mlp",
    "attention",
    "multimax",
    "rolling_w10",
    "rolling_w64",
    "rolling_w256",
)


def metrics(store: Store, logits: list[float]) -> dict:
    frame = pd.DataFrame(
        {
            "dataset": [r["source"] for r in store.rows],
            "label": [r["label"] for r in store.rows],
            "score": 1 / (1 + np.exp(-np.clip(logits, -60, 60))),
        }
    )
    return evaluate_binary_monitor(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--population", default="pilot")
    parser.add_argument("--backbone", choices=("prompted", "ntp"), required=True)
    parser.add_argument("--train-format", choices=("bf16", "fp8"), required=True)
    parser.add_argument("--eval-formats", nargs="+", default=["bf16", "fp8"])
    parser.add_argument("--eval-split", default="id")
    parser.add_argument("--architectures", nargs="+", default=list(ARCHITECTURES))
    parser.add_argument("--objectives", nargs="+", default=["hard", "soft"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-3)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    device = "cuda"
    root = Path(args.store_root) / args.population / args.backbone
    train = Store(root / "train", args.train_format)
    evals = {fmt: Store(root / args.eval_split, fmt) for fmt in args.eval_formats}
    out = (
        ROOT
        / "results/probe_generalization"
        / args.name
        / (f"{args.backbone}-{args.train_format}")
    )
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    runs = make_runs(
        args.architectures,
        args.objectives,
        args.seeds,
        train.meta["hidden"],
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )
    lengths = [r["length"] for r in train.rows]
    summary, started = {"checkpoints": {}}, time.time()
    for epoch in range(1, max(args.checkpoints) + 1):
        batches = length_batches(
            lengths, args.batch, seed=args.data_seed * 1000 + epoch
        )
        train_epoch(runs, train, batches, device)
        print(f"epoch {epoch} {time.time() - started:.0f}s", flush=True)
        if epoch not in args.checkpoints:
            continue
        checkpoint = {"epoch": epoch, "steps": len(runs[0].losses), "probes": {}}
        for fmt, store in evals.items():
            logits = score(runs, store, args.batch, device)
            ids = [r["id"] for r in store.rows]
            (out / f"logits_e{epoch}_{fmt}.json").write_text(
                json.dumps({"ids": ids, "logits": logits}) + "\n"
            )
            for run in runs:
                result = metrics(store, logits[run.name])
                entry = checkpoint["probes"].setdefault(
                    run.name,
                    {
                        "architecture": run.architecture,
                        "objective": run.objective,
                        "seed": run.seed,
                        "train_loss_last_epoch": float(
                            np.mean(run.losses[-len(batches) :])
                        ),
                    },
                )
                entry[fmt] = {"macro": result["macro"], "groups": result["groups"]}
        torch.save(
            {run.name: run.probe.state_dict() for run in runs},
            out / f"probes_e{epoch}.pt",
        )
        summary["checkpoints"][str(epoch)] = checkpoint
        summary["seconds"] = time.time() - started
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"checkpoint {epoch} written", flush=True)


if __name__ == "__main__":
    main()
