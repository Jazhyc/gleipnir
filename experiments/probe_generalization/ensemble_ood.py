"""Post-hoc OOD scores of every pooled probe at every checkpoint.

Added after the frozen OOD read (user decision 2026-10-25) so seed ensembles
of every architecture can be compared with the seed-averaged Gleipnir 4B
under any epoch rule (``ensemble.py``). One pass over the OOD store runs all
checkpoints' probe banks; per-row logits are saved locally. Every probe the
frozen read scored must reproduce its OOD pAUROC@20.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.probe_generalization.evaluate_ood import ROOT, report
from gleipnir.probes.store import Store
from gleipnir.probes.training import autocast, make_sweep, micro_batches, stream

# Same probes and activations; allows GPU/library rounding in near-tied rows.
TOLERANCE = 2e-3


def frozen_reads(path: Path) -> dict[tuple[int, str], float]:
    """``{(epoch, probe): OOD pAUROC@20}`` from the frozen read."""
    pooled = json.loads(path.read_text())["pooled"]
    out = {}
    for cell in pooled.values():
        for pick in [cell["selected"], *cell["families"].values()]:
            for name, result in pick["ood_per_probe"].items():
                out[(pick["epoch"], name)] = result["macro"]["pauroc_at_20"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--population", default="full")
    parser.add_argument("--backbone", choices=("prompted", "ntp"), required=True)
    parser.add_argument("--name", default="full01")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    results = ROOT / "results/probe_generalization" / args.name
    run = results / f"{args.backbone}-fp8"
    config = json.loads((run / "config.json").read_text())
    epochs = sorted(int(p.stem.split("_e")[1]) for p in run.glob("probes_e*.pt"))

    skip = 1 if args.backbone == "ntp" else 0
    store = Store(
        Path(args.store_root) / args.population / args.backbone / "ood",
        "fp8",
        skip=skip,
    )
    sweeps = {}
    for epoch in epochs:
        sweep = make_sweep(
            config["architectures"],
            config["objectives"],
            config["seeds"],
            store.meta["hidden"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            device=args.device,
        )
        state = torch.load(run / f"probes_e{epoch}.pt", map_location=args.device)
        if [tuple(s) for s in state["specs"]] != [
            (s.name, s.architecture) for s in sweep.specs
        ]:
            raise ValueError(f"probe order differs in {run}/probes_e{epoch}.pt")
        sweep.bank.load_state_dict(state["bank"])
        sweep.bank.eval()
        sweeps[epoch] = sweep
    names = [s.name for s in sweeps[epochs[0]].specs]

    lengths = [r["length"] for r in store.rows]
    order = sorted(range(len(store)), key=lambda i: lengths[i])
    batches = [
        chunk
        for begin in range(0, len(order), config["batch"])
        for chunk in micro_batches(
            order[begin : begin + config["batch"]], lengths, config["token_budget"]
        )
    ]
    logits = torch.zeros((len(epochs), len(store), len(names)))
    done = 0
    with torch.no_grad():
        for indices, x, mask in stream(store, batches, args.device):
            with autocast(args.device):
                for k, epoch in enumerate(epochs):
                    logits[k, indices] = sweeps[epoch].bank(x, mask).float().cpu()
            done += len(indices)
            if done % 1000 < len(indices):
                print(f"{args.backbone}: {done}/{len(store)} rows", flush=True)

    drift = {}
    for (epoch, name), expected in frozen_reads(
        results / f"ood_{args.backbone}.json"
    ).items():
        column = logits[epochs.index(epoch), :, names.index(name)].numpy()
        got = report(store.rows, column)["macro"]["pauroc_at_20"]
        drift[f"e{epoch}/{name}"] = got - expected
    worst = max(abs(v) for v in drift.values())
    out = results / "ensemble_ood"
    out.mkdir(exist_ok=True)
    np.savez_compressed(
        out / f"logits_{args.backbone}.npz",
        logits=logits.numpy().astype(np.float32),
        epochs=np.array(epochs),
        names=np.array(names),
        ids=np.array([r["id"] for r in store.rows]),
    )
    (out / f"parity_{args.backbone}.json").write_text(
        json.dumps({"max_abs_drift": worst, "drift": drift}, indent=2) + "\n"
    )
    print(f"{args.backbone}: frozen-read parity, worst |drift| {worst:.2e}")
    if worst > TOLERANCE:
        raise SystemExit(f"parity failed: {worst} > {TOLERANCE}")


if __name__ == "__main__":
    main()
