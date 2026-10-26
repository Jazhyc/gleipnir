"""Score the frozen NTP probe picks on SleightBench once, after a parity gate.

Picks are the probe study's frozen NTP selections (``selection_ntp.json``):
each objective's selected candidate and each family's ID-best candidate, all
5 seeds. No probe is retrained or reselected.

1. **Parity.** The picks score the ``parity_id`` rows from the re-built
   extraction; every logit must match the logit stored by the probe study
   within ``PARITY_TOLERANCE`` ID standard deviations of that probe.
2. **Single pass.** Only then are the ``stripped`` rows scored, once. Raw
   logits are written before any metric; an existing logits file refuses.
3. **Report** (``report``, CPU, separate from the GPU pass). SleightBench
   metrics (``sleightbench_eval`` ``view_report``,
   on logits) per seed, the seed median, the best-ID seed and the 5-seed
   mean-logit ensemble (descriptive), plus score-length rank correlations.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np
import torch

from gleipnir.data.monitoring import write_json
from gleipnir.probes.store import Store
from gleipnir.probes.training import make_sweep, score

EXPERIMENT = Path(__file__).parent
ROOT = EXPERIMENT.parents[1]
PROBES = ROOT / "results/probe_generalization/full01"
RESULTS = ROOT / "results/probe_sleightbench"
VIEW = "stripped"
PARITY_TOLERANCE = 0.05
HEADLINE = ("catch_rate", "auroc", "pauroc_at_20", "pair_win_rate")


def picks() -> list[dict]:
    """Unique (objective, architecture, epoch) picks with their roles."""
    selection = json.loads((PROBES / "selection_ntp.json").read_text())
    found: dict[tuple, dict] = {}
    for objective, cell in selection.items():
        roles = [("selected", cell["selected"])] + [
            (f"family:{f}", c) for f, c in cell["families"].items()
        ]
        for role, c in roles:
            key = (objective, c["architecture"], c["epoch"])
            entry = found.setdefault(
                key,
                {
                    "objective": objective,
                    "architecture": c["architecture"],
                    "epoch": c["epoch"],
                    "probes": c["probes"],
                    "id_seed_median": c["id_seed_median"],
                    "roles": [],
                },
            )
            entry["roles"].append(role)
    return list(found.values())


def score_store(store: Store, chosen: list[dict], device: str) -> dict[str, list]:
    """Logits of every picked probe on ``store``; other probes are dropped."""
    config = json.loads((PROBES / "ntp-fp8/config.json").read_text())
    wanted = defaultdict(set)
    for p in chosen:
        wanted[p["epoch"]].update(p["probes"])
    out = {}
    for epoch, names in sorted(wanted.items()):
        sweep = make_sweep(
            config["architectures"],
            config["objectives"],
            config["seeds"],
            store.hidden,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            device=device,
        )
        state = torch.load(PROBES / f"ntp-fp8/probes_e{epoch}.pt", map_location=device)
        if [tuple(s) for s in state["specs"]] != [
            (s.name, s.architecture) for s in sweep.specs
        ]:
            raise ValueError(f"probe order differs in probes_e{epoch}.pt")
        sweep.bank.load_state_dict(state["bank"])
        logits = score(sweep, store, config["batch"], device, config["token_budget"])
        out.update({f"{n}@e{epoch}": logits[n] for n in names})
    return out


def parity(store: Store, chosen: list[dict], device: str) -> dict:
    new = score_store(store, chosen, device)
    report = {"tolerance_id_sd": PARITY_TOLERANCE, "rows": len(store), "probes": {}}
    stored_by_epoch = {}
    for key, values in new.items():
        name, epoch = key.rsplit("@e", 1)
        if epoch not in stored_by_epoch:
            saved = json.loads(
                (PROBES / f"ntp-fp8/logits_e{epoch}_fp8.json").read_text()
            )
            stored_by_epoch[epoch] = (
                {i: k for k, i in enumerate(saved["ids"])},
                saved["logits"],
            )
        position, logits = stored_by_epoch[epoch]
        reference = np.asarray(logits[name], dtype=float)
        expected = reference[[position[r["id"]] for r in store.rows]]
        drift = np.abs(np.asarray(values) - expected) / reference.std()
        report["probes"][key] = {
            "max_drift_id_sd": float(drift.max()),
            "max_abs": float(np.abs(np.asarray(values) - expected).max()),
        }
    worst = max(p["max_drift_id_sd"] for p in report["probes"].values())
    report["max_drift_id_sd"] = worst
    report["passed"] = bool(worst <= PARITY_TOLERANCE)
    return report


def spearman(x, y) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def evaluate(pick: dict, logits: dict, rows: list[dict]) -> dict:
    from experiments.sleightbench_eval.run import view_report

    transcripts = json.loads(
        (ROOT / "experiments/sleightbench_eval/manifest.json").read_text()
    )["transcripts"]
    tokens = {r["id"]: r["input_tokens"] for r in rows}
    ids = [r["id"] for r in rows]
    labels = np.array([r["label"] for r in rows])
    length = np.log([r["input_tokens"] for r in rows])

    def one(values) -> dict:
        result = view_report(
            VIEW, dict(zip(ids, values, strict=True)), tokens, transcripts
        )
        values = np.asarray(values)
        result["length_spearman"] = {
            "attack": spearman(length[labels == 1], values[labels == 1]),
            "benign": spearman(length[labels == 0], values[labels == 0]),
        }
        return result

    columns = {n: logits[f"{n}@e{pick['epoch']}"] for n in pick["probes"]}
    per_seed = {n: one(v) for n, v in columns.items()}
    summary = json.loads((PROBES / "ntp-fp8/summary.json").read_text())
    id_entries = summary["checkpoints"][str(pick["epoch"])]["probes"]
    best_id = max(
        pick["probes"], key=lambda n: id_entries[n]["fp8"]["macro"]["pauroc_at_20"]
    )
    ensemble = np.mean([columns[n] for n in pick["probes"]], axis=0)
    return {
        **pick,
        "seed_median": {
            m: statistics.median(r["units"][m] for r in per_seed.values())
            for m in HEADLINE
        }
        | {
            "short_context_auroc": statistics.median(
                r["short_context"]["auroc"] for r in per_seed.values()
            ),
            "length_spearman_benign": statistics.median(
                r["length_spearman"]["benign"] for r in per_seed.values()
            ),
            "length_spearman_attack": statistics.median(
                r["length_spearman"]["attack"] for r in per_seed.values()
            ),
        },
        "best_id_seed": {"probe": best_id, **per_seed[best_id]},
        "ensemble_mean_logit": one(ensemble.tolist()),
        "per_seed": per_seed,
    }


def run_score(args: argparse.Namespace) -> None:
    out = RESULTS / args.name
    raw = out / "logits.json"
    if raw.exists():
        raise ValueError(f"{raw} exists; SleightBench is scored once")
    out.mkdir(parents=True, exist_ok=True)
    chosen = picks()
    root = Path(args.store_root) / "sleightbench/ntp"

    gate = parity(Store(root / "parity_id", "fp8", skip=1), chosen, args.device)
    write_json(out / "parity.json", gate)
    print(f"parity max drift {gate['max_drift_id_sd']:.4f} ID-SD", flush=True)
    if not gate["passed"]:
        raise SystemExit("parity gate failed; SleightBench rows not scored")

    store = Store(root / VIEW, "fp8", skip=1)
    logits = score_store(store, chosen, args.device)
    write_json(
        raw,
        {
            "ids": [r["id"] for r in store.rows],
            "input_tokens": [r["input_tokens"] for r in store.rows],
            "labels": [r["label"] for r in store.rows],
            "logits": logits,
        },
    )
    print(f"scored {len(store)} rows with {len(logits)} probes", flush=True)


def run_report(args: argparse.Namespace) -> None:
    out = RESULTS / args.name
    if not json.loads((out / "parity.json").read_text())["passed"]:
        raise ValueError("parity gate did not pass")
    raw = json.loads((out / "logits.json").read_text())
    rows = [
        {"id": i, "input_tokens": t, "label": y}
        for i, t, y in zip(raw["ids"], raw["input_tokens"], raw["labels"], strict=True)
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        evaluated = list(
            pool.map(evaluate, picks(), repeat(raw["logits"]), repeat(rows))
        )
    report = {
        "view": VIEW,
        "selection": "probe_generalization full01 selection_ntp.json (frozen)",
        "picks": evaluated,
    }
    write_json(out / "report.json", report)
    for p in evaluated:
        m = p["seed_median"]
        print(
            f"{p['objective']} {p['architecture']} e{p['epoch']} "
            f"{','.join(p['roles'])}: catch {m['catch_rate']:.3f} "
            f"AUROC {m['auroc']:.3f} pAUROC@20 {m['pauroc_at_20']:.3f} "
            f"pair wins {m['pair_win_rate']:.3f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("score", help="GPU: parity gate, then logits")
    command.add_argument("--store-root", required=True)
    command.add_argument("--name", required=True)
    command.add_argument("--device", default="cuda")
    command = commands.add_parser("report", help="CPU: metrics from saved logits")
    command.add_argument("--name", required=True)
    command.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_score(args) if args.command == "score" else run_report(args)


if __name__ == "__main__":
    main()
