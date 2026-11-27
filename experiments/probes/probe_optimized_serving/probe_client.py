"""Score rows against the compiled probe server and time the pass.

Rows go to vLLM's ``/classify`` route in
sequential 128-row batches at concurrency 128, as in the v0.3 benchmark. One
excluded warmup batch runs first. Each response carries the probe pooler's
flat logit vector (variant x epoch x probe, layout in the server's probe
receipt) and the prompt token count, which must match the Transformers
tokenizer. Rows are bare trajectories (NTP rendering) or, with ``input`` and
``tokens`` fields, rows prepared by ``prepare_27b.py``. Logits go to
``logits.npz`` and timing to ``timing.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np

BATCH = 128


async def score_batch(port: int, rows: list[dict]) -> tuple:
    """One fresh client per batch, as in ``gleipnir.evaluation.http_score``:
    keep-alive connections reused across batches can hit server-closed
    sockets (``httpx.ReadError``)."""
    client = httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}",
        trust_env=False,
        timeout=1800,
        limits=httpx.Limits(max_connections=BATCH),
    )

    async def one(row: dict) -> list[float]:
        response = await client.post(
            "/classify",
            json={
                "model": "monitor",
                "input": row["input"],
                "add_special_tokens": False,
                "use_activation": False,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if payload["usage"]["prompt_tokens"] != row["tokens"]:
            raise ValueError(
                f"token mismatch in {row['id']}: server "
                f"{payload['usage']['prompt_tokens']}, tokenizer {row['tokens']}"
            )
        return payload["data"][0]["probs"]

    async with client:
        before = time.perf_counter()
        values = await asyncio.gather(*(one(r) for r in rows))
        return values, time.perf_counter() - before


async def run(rows: list[dict], port: int) -> tuple[np.ndarray, dict]:
    await score_batch(port, rows[:BATCH])  # warmup, excluded
    values, times = [], []
    for start in range(0, len(rows), BATCH):
        batch, seconds = await score_batch(port, rows[start : start + BATCH])
        values.extend(batch)
        times.append(seconds)
        print(f"batch {start // BATCH} {seconds:.2f}s", flush=True)
    logits = np.array(values, dtype=np.float32)
    if not np.isfinite(logits).all():
        raise ValueError("nonfinite probe logits")
    tokens = sum(r["tokens"] for r in rows)
    timing = {
        "rows": len(rows),
        "tokens": tokens,
        "seconds": sum(times),
        "batch_seconds": times,
        "tokens_per_second": tokens / sum(times),
        "rows_per_second": len(rows) / sum(times),
    }
    return logits, timing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, help="jsonl with id, trajectory")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--receipt", required=True, help="server probe receipt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    rows = [json.loads(line) for line in open(args.rows)]
    rows = rows[: args.limit] if args.limit else rows
    for row in rows:
        if "input" not in row:
            row["input"] = row["trajectory"]
            ids = tokenizer(row["input"], add_special_tokens=False)["input_ids"]
            row["tokens"] = len(ids)
    receipt = json.loads(Path(args.receipt).read_text())
    logits, timing = asyncio.run(run(rows, args.port))
    shape = (
        len(rows),
        len(receipt["variants"]),
        len(receipt["epochs"]),
        len(receipt["probes"]),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "logits.npz",
        logits=logits.reshape(shape),
        ids=np.array([r["id"] for r in rows]),
        tokens=np.array([r["tokens"] for r in rows]),
        variants=np.array(receipt["variants"]),
        epochs=np.array(receipt["epochs"]),
        names=np.array(receipt["probes"]),
    )
    summary = receipt | {"probes": len(receipt["probes"])}
    timing.update(rows_file=args.rows, receipt=summary)
    (out / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps({k: v for k, v in timing.items() if k != "batch_seconds"}))


if __name__ == "__main__":
    main()
