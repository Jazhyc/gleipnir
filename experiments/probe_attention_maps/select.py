"""Frozen row selection for the attention-map study (CPU, saved logits only).

Probes: the NTP-backbone attention family, hard and soft, epoch 8 (the frozen
ID pick of that family for both objectives), seeds 0-4. Each probe is
thresholded at its own 5% ID false-positive rate. Sets over the OOD suite:

- ``soft_fixes``: all 5 hard seeds wrong and all 5 soft seeds right;
- ``hard_fixes``: the reverse;
- ``both_right_pos`` / ``both_missed_pos``: positives all 10 probes get right
  / all 10 miss.

The rank gain of a row is the seed-mean OOD percentile rank of its soft logit
minus that of its hard logit (positive: soft moves the row toward "attack").
Example rows for token-level maps, chosen deterministically: per source, the
``PER_SOURCE`` soft fixes with the largest rank gain; the ``REVERSE`` hard fixes
with the most negative gain; and per source the both-right positive with the
median rank gain, as a reference for what an attack the hard probe already
catches looks like.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/probe_generalization/full01"
DATA = ROOT / "data/probe_generalization/full"
EPOCH = 8
SEEDS = range(5)
OBJECTIVES = ("hard", "soft")
ID_FPR = 0.05
PER_SOURCE = 2
REVERSE = 4
PROBES = [f"attention-{o}-s{s}" for o in OBJECTIVES for s in SEEDS]


def percentile_ranks(x: np.ndarray) -> np.ndarray:
    """Average-tie percentile rank in [0, 1] along the last axis."""
    out = np.empty(x.shape)
    for i, row in enumerate(x):
        _, inverse, counts = np.unique(row, return_inverse=True, return_counts=True)
        below = np.concatenate([[0], np.cumsum(counts)[:-1]])
        out[i] = (below + (counts - 1) / 2)[inverse] / max(len(row) - 1, 1)
    return out


def select(
    id_logits: np.ndarray,
    id_labels: np.ndarray,
    ood_logits: np.ndarray,
    ood_labels: np.ndarray,
    sources: np.ndarray,
) -> dict:
    """``*_logits``: ``[10 probes (hard s0-4, soft s0-4), rows]``."""
    thresholds = np.quantile(id_logits[:, id_labels == 0], 1 - ID_FPR, axis=1)
    correct = (ood_logits > thresholds[:, None]) == ood_labels.astype(bool)
    hard, soft = correct[:5], correct[5:]
    pos = ood_labels == 1
    sets = {
        "soft_fixes": (~hard).all(0) & soft.all(0),
        "hard_fixes": hard.all(0) & (~soft).all(0),
        "both_right_pos": pos & correct.all(0),
        "both_missed_pos": pos & (~correct).all(0),
    }
    ranks = percentile_ranks(ood_logits)
    gain = ranks[5:].mean(0) - ranks[:5].mean(0)
    examples = []
    for source in sorted(set(sources)):
        rows = np.flatnonzero(sets["soft_fixes"] & (sources == source))
        for i in rows[np.argsort(-gain[rows], kind="stable")][:PER_SOURCE]:
            examples.append((int(i), "soft_fix"))
        rows = np.flatnonzero(sets["both_right_pos"] & (sources == source))
        if len(rows):
            mid = rows[np.argsort(gain[rows], kind="stable")][len(rows) // 2]
            examples.append((int(mid), "both_right_reference"))
    rows = np.flatnonzero(sets["hard_fixes"])
    for i in rows[np.argsort(gain[rows], kind="stable")][:REVERSE]:
        examples.append((int(i), "hard_fix"))
    return {
        "thresholds": thresholds,
        "sets": sets,
        "rank_gain": gain,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="maps01")
    args = parser.parse_args()
    saved = json.loads((SOURCE / f"ntp-fp8/logits_e{EPOCH}_fp8.json").read_text())
    id_rows = {r["id"]: r for r in map(json.loads, (DATA / "id.jsonl").open())}
    id_labels = np.array([id_rows[i]["label"] for i in saved["ids"]])
    id_logits = np.array([saved["logits"][p] for p in PROBES], dtype=float)

    ood = np.load(SOURCE / "ensemble_ood/logits_ntp.npz")
    names = [str(n) for n in ood["names"]]
    k = [int(e) for e in ood["epochs"]].index(EPOCH)
    ood_logits = np.stack([ood["logits"][k, :, names.index(p)] for p in PROBES])
    ood_logits = ood_logits.astype(float)
    ids = [str(i) for i in ood["ids"]]
    ood_rows = {r["id"]: r for r in map(json.loads, (DATA / "ood.jsonl").open())}
    rows = [ood_rows[i] for i in ids]
    labels = np.array([r["label"] for r in rows])
    sources = np.array([r["source"] for r in rows])

    out = select(id_logits, id_labels, ood_logits, labels, sources)
    run = ROOT / "results/probe_attention_maps" / args.name
    run.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run / "selection.npz",
        ids=np.array(ids),
        probes=np.array(PROBES),
        ood_logits=ood_logits.astype(np.float32),
        thresholds=out["thresholds"],
        rank_gain=out["rank_gain"],
        **{f"set_{k}": v for k, v in out["sets"].items()},
    )
    summary = {
        "probes": PROBES,
        "epoch": EPOCH,
        "id_fpr": ID_FPR,
        "thresholds": dict(zip(PROBES, out["thresholds"].tolist(), strict=True)),
        "set_sizes": {
            k: {
                "total": int(v.sum()),
                "positives": int((v & (labels == 1)).sum()),
                "by_source": {
                    s: int((v & (sources == s)).sum()) for s in sorted(set(sources))
                },
            }
            for k, v in out["sets"].items()
        },
        "examples": [
            {
                "index": i,
                "id": ids[i],
                "kind": kind,
                "source": rows[i]["source"],
                "label": rows[i]["label"],
                "rank_gain": float(out["rank_gain"][i]),
                "hard_logits": ood_logits[:5, i].tolist(),
                "soft_logits": ood_logits[5:, i].tolist(),
            }
            for i, kind in out["examples"]
        ],
    }
    (run / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["set_sizes"], indent=1))
    for e in summary["examples"]:
        print(e["kind"], e["source"], e["label"], f"{e['rank_gain']:+.3f}", e["id"])


if __name__ == "__main__":
    main()
