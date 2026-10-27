"""Compare the frozen full01 NTP probe picks: BF16 pipeline vs optimized stack.

For each objective's selected cell and each family's ID-best candidate
(``full01/selection_ntp.json``), report the seed-median source-macro
pAUROC@20 on ID and OOD under both activation sources, with the paired
difference. The BF16 references are the full01 logits (ID: the sweep's
checkpoint logits; OOD: the post-hoc all-probe scoring, whose frozen picks
reproduce the frozen OOD read exactly). The selected cells also get a paired,
source-stratified bootstrap of the difference. Nothing is selected here.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from experiments.probe_generalization.evaluate_ood import report

ROOT = Path(__file__).resolve().parents[2]
FULL01 = ROOT / "results/probe_generalization/full01"
BOOTSTRAP = 200


def rows_of(split: str) -> dict[str, dict]:
    path = ROOT / "data/probe_generalization/full" / f"{split}.jsonl"
    return {
        r["id"]: {"source": r["source"], "label": r["label"]}
        for r in map(json.loads, path.open())
    }


def reference(split: str, epoch: int) -> tuple[list[str], dict[str, np.ndarray]]:
    if split == "id":
        data = json.loads((FULL01 / f"ntp-fp8/logits_e{epoch}_fp8.json").read_text())
        return data["ids"], {k: np.array(v) for k, v in data["logits"].items()}
    data = np.load(FULL01 / "ensemble_ood/logits_ntp.npz")
    k = list(data["epochs"]).index(epoch)
    return list(data["ids"]), {
        str(n): data["logits"][k][:, j] for j, n in enumerate(data["names"])
    }


def optimized(run: Path, split: str, epoch: int, variant: str) -> tuple:
    data = np.load(run / split / "logits.npz")
    v = list(data["variants"]).index(variant)
    k = list(data["epochs"]).index(epoch)
    return list(data["ids"]), {
        str(n): data["logits"][:, v, k, j] for j, n in enumerate(data["names"])
    }


def macro(rows: list[dict], logits: np.ndarray) -> float:
    return report(rows, logits)["macro"]["pauroc_at_20"]


def seed_median(rows, logits: dict, probes: list[str], index=None) -> float:
    if index is not None:
        rows = [rows[i] for i in index]
        logits = {p: logits[p][index] for p in probes}
    return statistics.median(macro(rows, logits[p]) for p in probes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="results dir with id/, ood/")
    parser.add_argument("--variant", default="fp8")
    args = parser.parse_args()
    run = Path(args.run)
    selection = json.loads((FULL01 / "selection_ntp.json").read_text())
    picks = []
    for objective, cell in selection.items():
        picks.append((objective, "selected", cell["selected"]))
        picks.extend((objective, f, c) for f, c in cell["families"].items())
    result = {"variant": args.variant, "picks": []}
    rng = np.random.default_rng(0)
    for split in ("id", "ood"):
        meta = rows_of(split)
        cache = {}
        for objective, name, pick in picks:
            epoch = pick["epoch"]
            if epoch not in cache:
                ref_ids, ref = reference(split, epoch)
                opt_ids, opt = optimized(run, split, epoch, args.variant)
                order = {i: k for k, i in enumerate(opt_ids)}
                missing = set(ref_ids) - set(order)
                if missing:
                    raise ValueError(f"{split}: {len(missing)} rows not scored")
                align = np.array([order[i] for i in ref_ids])
                opt = {k: v[align] for k, v in opt.items()}
                rows = [meta[i] for i in ref_ids]
                cache[epoch] = (rows, ref, opt)
            rows, ref, opt = cache[epoch]
            probes = pick["probes"]
            entry = {
                "split": split,
                "objective": objective,
                "pick": name,
                "architecture": pick["architecture"],
                "epoch": epoch,
                "bf16": seed_median(rows, ref, probes),
                "optimized": seed_median(rows, opt, probes),
                "logit_corr": float(
                    np.mean([np.corrcoef(ref[p], opt[p])[0, 1] for p in probes])
                ),
            }
            entry["delta"] = entry["optimized"] - entry["bf16"]
            if name == "selected":
                by_source = {}
                for k, r in enumerate(rows):
                    by_source.setdefault(r["source"], []).append(k)
                deltas = []
                for _ in range(BOOTSTRAP):
                    index = np.concatenate(
                        [rng.choice(v, len(v)) for v in by_source.values()]
                    )
                    deltas.append(
                        seed_median(rows, opt, probes, index)
                        - seed_median(rows, ref, probes, index)
                    )
                entry["delta_ci95"] = [
                    float(np.percentile(deltas, 2.5)),
                    float(np.percentile(deltas, 97.5)),
                ]
            result["picks"].append(entry)
            ci = entry.get("delta_ci95")
            print(
                f"{split} {objective:4s} {name:12s} {pick['architecture']:24s} "
                f"e{epoch:<2d} bf16 {entry['bf16']:.4f} opt {entry['optimized']:.4f} "
                f"delta {entry['delta']:+.4f} corr {entry['logit_corr']:.3f}"
                + (f" CI [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""),
                flush=True,
            )
    (run / f"comparison_{args.variant}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
