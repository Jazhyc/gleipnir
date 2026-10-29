"""Compare optimized-stack block-15 activations with the BF16 extraction.

Both are post-block-15 residual streams on the same diagnostic rows. For each
row the token counts must match. Reported per row and pooled over tokens
(excluding the attention sink): relative L2 error, cosine similarity, and,
as a yardstick, the same statistics for the study's FP8 storage round trip of
the BF16 activations (the error that passed the FP8 gate).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gleipnir.probes import fp8
from gleipnir.probes.store import Store


def stats(test: torch.Tensor, ref: torch.Tensor) -> dict:
    diff = (test - ref).norm(dim=-1)
    norm = ref.norm(dim=-1)
    cosine = torch.nn.functional.cosine_similarity(test, ref, dim=-1)
    return {
        "relative_l2_mean": float((diff / norm).mean()),
        "relative_l2_p95": float(torch.quantile(diff / norm, 0.95)),
        "pooled_relative_l2": float(diff.norm() / norm.norm()),
        "cosine_mean": float(cosine.mean()),
        "cosine_min": float(cosine.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-store", required=True)
    parser.add_argument("--optimized", required=True, help="dir of <id>.pt captures")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    store = Store(Path(args.bf16_store), "bf16", skip=1)
    per_row, tests, refs, stored = [], [], [], []
    for i, row in enumerate(store.rows):
        ref = store.load(i, "cuda").float()
        test = torch.load(Path(args.optimized) / f"{row['id']}.pt").cuda().float()[1:]
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
    pooled = {
        "optimized": stats(torch.cat(tests), torch.cat(refs)),
        "fp8_storage": stats(torch.cat(stored), torch.cat(refs)),
    }
    by_source = {}
    for source in sorted({r["source"] for r in per_row}):
        mine = [r for r in per_row if r["source"] == source]
        by_source[source] = {
            "rows": len(mine),
            "relative_l2_mean": float(
                np.mean([r["optimized"]["relative_l2_mean"] for r in mine])
            ),
        }
    result = {"rows": len(per_row), "pooled": pooled, "by_source": by_source}
    Path(args.out).write_text(
        json.dumps(result | {"per_row": per_row}, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
