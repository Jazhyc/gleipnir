"""Score saved block-15 activations with one full01 probe checkpoint (diagnostic).

Validates the probe pooler's plumbing: activations captured by eager Lens
(``<id>.pt``, sink included) or a BF16 probe store go through the same sink
drop and blockwise FP8 round trip as the pooler, then the bank. Output rows
follow the input order: ``{ids, logits [rows, probes], names}``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.probe_optimized_serving.capture import Chunk, load_sweeps
from gleipnir.probes import fp8
from gleipnir.probes.store import Store
from gleipnir.probes.training import score

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--lens-dir")
    source.add_argument("--bf16-store")
    parser.add_argument(
        "--probe-run", default="results/probe_generalization/full01/ntp-fp8"
    )
    parser.add_argument("--epoch", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    ids = Path(args.ids).read_text().split()
    if args.lens_dir:
        values = [torch.load(Path(args.lens_dir) / f"{i}.pt")[1:] for i in ids]
    else:
        store = Store(Path(args.bf16_store), "bf16", skip=1)
        index = {r["id"]: k for k, r in enumerate(store.rows)}
        values = [store.load(index[i]) for i in ids]
    values = [fp8.dequantize(*fp8.quantize(v.cuda())).cpu() for v in values]
    sweeps, names = load_sweeps(ROOT / args.probe_run, [args.epoch])
    logits = score(sweeps[args.epoch], Chunk(values), 4, "cuda", 16384)
    np.savez(
        args.out,
        ids=np.array(ids),
        logits=np.array([logits[n] for n in names]).T,
        names=np.array(names),
    )


if __name__ == "__main__":
    main()
