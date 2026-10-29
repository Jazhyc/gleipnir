"""27B activation diagnostic and plumbing check on the 64 diagnostic OOD rows.

Inputs: the Pod's BF16 Transformers extraction at block 31 (a probe store with
``bf16`` and ``fp8`` formats), the optimized server's dumped kept states
(``<key>.pt``, already trimmed to the kept span) and the prepared rows.

- **Activations.** Per-token relative L2 error and cosine of the optimized
  states against BF16, with the FP8 storage round trip as a yardstick
  (``diag_compare.stats``).
- **Plumbing.** The attention picks' probes score the extraction's FP8 store;
  their logits must match the Lambda reference (``ensemble_ood``) for the
  same rows, at mean correlation of at least 0.99.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.probe_optimized_serving.compare import reference
from experiments.probe_optimized_serving.diag_compare import stats
from gleipnir.probes import fp8
from gleipnir.probes.store import Store
from gleipnir.probes.training import make_sweep, score

ROOT = Path(__file__).resolve().parents[2]
MIN_CORRELATION = 0.99


def attention_picks(sweep: Path, backbone: str) -> dict[int, list[str]]:
    selection = json.loads((sweep / f"selection_{backbone}.json").read_text())
    picks: dict[int, list[str]] = {}
    for cell in selection.values():
        pick = cell["families"]["attention"]
        picks.setdefault(pick["epoch"], []).extend(pick["probes"])
    return picks


def plumbing(store: Store, sweep: Path, backbone: str) -> dict:
    config = json.loads((sweep / f"{backbone}-fp8/config.json").read_text())
    ids = [r["id"] for r in store.rows]
    result = {}
    for epoch, probes in attention_picks(sweep, backbone).items():
        bank = make_sweep(
            config["architectures"],
            config["objectives"],
            config["seeds"],
            store.hidden,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            device="cuda",
        )
        state = torch.load(
            sweep / f"{backbone}-fp8/probes_e{epoch}.pt", map_location="cuda"
        )
        bank.bank.load_state_dict(state["bank"])
        got = score(bank, store, 4, "cuda", config["token_budget"])
        ref_ids, ref = reference("ood", epoch, sweep, backbone)
        index = [ref_ids.index(i) for i in ids]
        for probe in probes:
            result[f"e{epoch}/{probe}"] = float(
                np.corrcoef(np.array(got[probe]), ref[probe][index])[0, 1]
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=("ntp", "prompted"), required=True)
    parser.add_argument("--store", required=True, help="BF16 extraction OOD store")
    parser.add_argument("--dump", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    skip = 1 if args.backbone == "ntp" else 0
    keys = {r["id"]: r["key"] for r in map(json.loads, open(args.rows))}
    store = Store(Path(args.store), "bf16", skip=skip)
    per_row, tests, refs, stored = [], [], [], []
    for i, row in enumerate(store.rows):
        ref = store.load(i, "cuda").float()
        test = torch.load(Path(args.dump) / f"{keys[row['id']]}.pt").cuda().float()
        if test.shape != ref.shape:
            raise ValueError(f"shape mismatch in {row['id']}: {test.shape} {ref.shape}")
        rounded = fp8.dequantize(*fp8.quantize(ref)).float()
        per_row.append(
            {
                "id": row["id"],
                "source": row["source"],
                "tokens": len(ref),
                "optimized": stats(test, ref),
                "fp8_storage": stats(rounded, ref),
            }
        )
        tests.append(test.cpu())
        refs.append(ref.cpu())
        stored.append(rounded.cpu())
    correlations = plumbing(
        Store(Path(args.store), "fp8", skip=skip), ROOT / args.sweep, args.backbone
    )
    result = {
        "backbone": args.backbone,
        "rows": len(per_row),
        "pooled": {
            "optimized": stats(torch.cat(tests), torch.cat(refs)),
            "fp8_storage": stats(torch.cat(stored), torch.cat(refs)),
        },
        "plumbing_correlation": correlations,
        "plumbing_mean": float(np.mean(list(correlations.values()))),
    }
    Path(args.out).write_text(
        json.dumps(result | {"per_row": per_row}, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    if result["plumbing_mean"] < MIN_CORRELATION:
        raise SystemExit(f"plumbing check failed: {result['plumbing_mean']:.4f}")


if __name__ == "__main__":
    main()
