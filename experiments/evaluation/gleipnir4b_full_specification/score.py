"""Score Gleipnir 4B on the full-specification workloads of the 27B.

Runs against an already started server on the Pod: stock BF16 ``vllm serve``
(``--backend bf16``) or the released optimized server (``--backend
optimized``). Writes one row per prompt (id, score, margin); metrics are
computed by ``compare.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import math
from pathlib import Path

from experiments.release_serving.targets import ROOT
from gleipnir.data.monitoring import read_rows, write_json, write_rows

WORKLOADS = {
    "id": ("data/optimized_gap/teacher_id_workload.jsonl", 3012),
    "ood": ("data/optimized_gap/teacher_ood_workload.jsonl", 6395),
}
BATCH_ROWS = 128


async def score_rows(batch: list[dict], backend: str, port: int) -> list[dict]:
    if backend == "bf16":
        from experiments.release_serving.stock_vllm import classify

        scored = await classify(batch, port)
        return [{**r, "margin": r["logits"][1] - r["logits"][0]} for r in scored]
    from gleipnir.evaluation.http_score import score_batch

    config = {"port": port, "concurrency": 128, "timeout_seconds": 600}
    scored, _ = await score_batch(batch, config)
    return scored


def check(rows: list[dict], workload: list[dict], expected: int) -> None:
    """Require one finite score in [0, 1] for every workload row."""
    ids = [r["id"] for r in rows]
    if len(ids) != expected or set(ids) != {r["id"] for r in workload}:
        raise ValueError("scored rows do not cover the workload")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate scored rows")
    for row in rows:
        score = float(row["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"invalid score for {row['id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("bf16", "optimized"), required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=sorted(WORKLOADS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for split in args.splits or sorted(WORKLOADS):
        path, expected = WORKLOADS[split]
        workload = read_rows(ROOT / path)
        scored: list[dict] = []
        for start in range(0, len(workload), BATCH_ROWS):
            batch = workload[start : start + BATCH_ROWS]
            scored += asyncio.run(score_rows(batch, args.backend, args.port))
        check(scored, workload, expected)
        write_rows(
            args.output / f"{split}.jsonl",
            [
                {"id": r["id"], "score": r["score"], "margin": r["margin"]}
                for r in scored
            ],
        )
        receipt = {"backend": args.backend, "split": split, "rows": len(scored)}
        write_json(args.output / f"{split}_receipt.json", receipt)
        print("scored", receipt, flush=True)


if __name__ == "__main__":
    main()
