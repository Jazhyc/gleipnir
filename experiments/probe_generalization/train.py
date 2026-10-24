"""Train the pooled-probe sweep for one backbone and storage format.

All architectures, objectives and seeds train together from one shared batch
stream (:mod:`gleipnir.probes.training`). At every checkpoint epoch, each
probe scores the evaluation split in every requested format; logits, metrics
and probe weights are written under ``results/probe_generalization/<name>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Variable-length batches next to a large resident store fragment the CUDA
# cache; expandable segments avoid near-OOM flush-and-retry allocations.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from gleipnir.evaluation.metrics import evaluate_binary_monitor  # noqa: E402
from gleipnir.probes.store import ResidentStore, Store  # noqa: E402
from gleipnir.probes.training import (  # noqa: E402
    length_batches,
    make_sweep,
    score,
    train_epoch,
)

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
# Full run: the pilot family plus the AlphaEvolve probe over its unreported
# penalty weights (l1 x ortho), selected on ID like the window widths.
FULL_ARCHITECTURES = (
    *ARCHITECTURES,
    *(
        f"alphaevolve_l{l1}_o{ortho}"
        for l1 in ("1e-06", "1e-05")
        for ortho in ("0.001", "0.01")
    ),
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
    parser.add_argument(
        "--eval-formats",
        nargs="+",
        help="default both: every probe also scores the precision it was not "
        "trained on (FP8 gate a and the cross-precision transfer grid)",
    )
    parser.add_argument("--eval-split", default="id")
    parser.add_argument("--architectures", nargs="+")
    parser.add_argument("--objectives", nargs="+", default=["hard", "soft"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--token-budget",
        type=int,
        help="cap padded tokens per forward (gradient accumulation within batches)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-3)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument(
        "--resident",
        choices=("gpu", "cpu", "none"),
        default="gpu",
        help="hold the training rows in GPU or pinned CPU memory, or stream them",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        help="continue an existing sweep from this saved checkpoint epoch "
        "(probe weights, AdamW state and per-epoch data order are restored)",
    )
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.architectures is None:
        pilot = args.population == "pilot"
        args.architectures = list(ARCHITECTURES if pilot else FULL_ARCHITECTURES)

    device = "cuda"
    root = Path(args.store_root) / args.population / args.backbone
    # NTP rows start at sequence position 0, the attention sink (Qwen has no
    # BOS); the prompted span starts after the instruction prefix.
    skip = 1 if args.backbone == "ntp" else 0
    train = Store(root / "train", args.train_format, skip=skip)
    if args.resident != "none":
        loaded = time.time()
        train = ResidentStore(train, "cuda" if args.resident == "gpu" else "cpu")
        print(f"resident {args.resident} {time.time() - loaded:.0f}s", flush=True)
    if args.eval_formats is None:
        args.eval_formats = ["bf16", "fp8"]
    # Evaluation rows are read at every checkpoint: hold them in pinned CPU memory.
    loaded = time.time()
    evals = {
        fmt: ResidentStore(Store(root / args.eval_split, fmt, skip=skip), "cpu")
        for fmt in args.eval_formats
    }
    print(f"eval resident cpu {time.time() - loaded:.0f}s", flush=True)
    out = (
        ROOT
        / "results/probe_generalization"
        / args.name
        / (f"{args.backbone}-{args.train_format}")
    )
    out.mkdir(parents=True, exist_ok=args.resume_from is not None)
    config_name = (
        "config.json"
        if args.resume_from is None
        else f"config_resume_e{args.resume_from}.json"
    )
    (out / config_name).write_text(json.dumps(vars(args), indent=2) + "\n")
    sweep = make_sweep(
        args.architectures,
        args.objectives,
        args.seeds,
        train.meta["hidden"],
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )
    lengths = [r["length"] for r in train.rows]
    summary, started, first = {"checkpoints": {}}, time.time(), 1
    if args.resume_from is not None:
        state = torch.load(out / f"probes_e{args.resume_from}.pt", map_location=device)
        if [tuple(x) for x in state["specs"]] != [
            (s.name, s.architecture) for s in sweep.specs
        ]:
            raise ValueError("resumed sweep has a different probe order")
        sweep.bank.load_state_dict(state["bank"])
        sweep.optimizer.load_state_dict(
            torch.load(out / f"optimizer_e{args.resume_from}.pt", map_location=device)
        )
        summary = json.loads((out / "summary.json").read_text())
        summary["resumed_from"] = args.resume_from
        first = args.resume_from + 1
    for epoch in range(first, max(args.checkpoints) + 1):
        batches = length_batches(
            lengths, args.batch, seed=args.data_seed * 1000 + epoch
        )
        train_epoch(sweep, train, batches, device, args.token_budget)
        print(f"epoch {epoch} {time.time() - started:.0f}s", flush=True)
        if epoch not in args.checkpoints:
            continue
        checkpoint = {"epoch": epoch, "steps": epoch * len(batches), "probes": {}}
        for fmt, store in evals.items():
            logits = score(sweep, store, args.batch, device, args.token_budget)
            ids = [r["id"] for r in store.rows]
            (out / f"logits_e{epoch}_{fmt}.json").write_text(
                json.dumps({"ids": ids, "logits": logits}) + "\n"
            )
            for run in sweep.specs:
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
            {
                "specs": [(s.name, s.architecture) for s in sweep.specs],
                "bank": sweep.bank.state_dict(),
            },
            out / f"probes_e{epoch}.pt",
        )
        torch.save(sweep.optimizer.state_dict(), out / f"optimizer_e{epoch}.pt")
        summary["checkpoints"][str(epoch)] = checkpoint
        summary["seconds"] = time.time() - started
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"checkpoint {epoch} written", flush=True)


if __name__ == "__main__":
    main()
