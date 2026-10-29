"""Time the frozen OOD workload against a running monitor server.

The method matches the v0.3 benchmark: sequential 128-row batches at
concurrency 128, with throughput taken as tokens over summed batch wall time.
One excluded warmup batch runs first. ``--rendering ntp`` sends the bare
trajectories, with token counts taken from the Transformers tokenizer;
``prompted`` sends the frozen Gleipnir student-prompt workload unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from gleipnir.evaluation.http_score import score_batch

ROOT = Path(__file__).resolve().parents[2]
PROMPTED = ROOT / "data/teacher-disagreement-strict-ood01/workload.jsonl"
NTP = ROOT / "data/probe_generalization/full/ood.jsonl"
CONFIG = {"port": 8010, "timeout_seconds": 1800, "concurrency": 128}
BATCH = 128


def workload(rendering: str, snapshot: str) -> list[dict]:
    if rendering == "prompted":
        return [json.loads(line) for line in PROMPTED.open()]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    rows = []
    for line in NTP.open():
        row = json.loads(line)
        ids = tokenizer(row["trajectory"], add_special_tokens=False)["input_ids"]
        rows.append(
            {
                "id": row["id"],
                "prompt": row["trajectory"],
                "prompt_sha256": row["trajectory_sha256"],
                "prompt_tokens": len(ids),
            }
        )
    return rows


async def run(rows: list[dict], port: int) -> dict:
    config = CONFIG | {"port": port}
    await score_batch(rows[:BATCH], config)  # warmup, excluded
    times = []
    for start in range(0, len(rows), BATCH):
        _, seconds = await score_batch(rows[start : start + BATCH], config)
        times.append(seconds)
        print(f"batch {start // BATCH} {seconds:.2f}s", flush=True)
    tokens = sum(r["prompt_tokens"] for r in rows)
    return {
        "rows": len(rows),
        "tokens": tokens,
        "seconds": sum(times),
        "batch_seconds": times,
        "tokens_per_second": tokens / sum(times),
        "rows_per_second": len(rows) / sum(times),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendering", choices=("ntp", "prompted"), required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    result = asyncio.run(run(workload(args.rendering, args.snapshot), args.port))
    result.update(label=args.label, rendering=args.rendering)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "batch_seconds"}))


if __name__ == "__main__":
    main()
