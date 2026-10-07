"""Continuous monitor requests with prompt-bound, append-only recovery receipts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from gleipnir.serving.monitor_score import ENDPOINT, validate_score_response


def recovery_records(
    checkpoint: Path, header: dict, rows: list[dict]
) -> tuple[dict[int, dict], int]:
    """Validate a journal before reusing completed responses or repairing its tail."""
    if not checkpoint.exists():
        return {}, 0
    data = checkpoint.read_bytes()
    lines = data.splitlines(keepends=True)
    if not lines or not lines[0].endswith(b"\n") or json.loads(lines[0]) != header:
        raise ValueError("continuous checkpoint contract changed")
    records, valid_bytes = {}, len(lines[0])
    for position, line in enumerate(lines[1:], 1):
        if not line.endswith(b"\n"):
            if position != len(lines) - 1:
                raise ValueError("invalid checkpoint interior")
            break
        entry = json.loads(line)
        index, value = entry["index"], entry["value"]
        if type(index) is not int or not 0 <= index < len(rows) or index in records:
            raise ValueError("duplicate or invalid checkpoint index")
        row = rows[index]
        if (value["id"], value["prompt_sha256"], value["prompt_tokens"]) != (
            row["id"],
            row["prompt_sha256"],
            row["prompt_tokens"],
        ):
            raise ValueError("continuous checkpoint prompt identity changed")
        validate_score_response(value, row["prompt_tokens"])
        records[index] = value
        valid_bytes += len(line)
    # Only an incomplete final line can be discarded, after validating the prefix.
    if valid_bytes != len(data):
        with checkpoint.open("r+b") as handle:
            handle.truncate(valid_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    return records, valid_bytes


async def continuous_score_trial(
    rows: list[dict],
    concurrency: int,
    settings: dict,
    *,
    checkpoint: Path,
    contract: dict,
    progress: Callable[[int, int], None] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[list[dict], float, dict[str, Any]]:
    """Refill request slots immediately and persist responses without group barriers.

    A single background writer flushes and fsyncs completion-order JSONL records.
    Recovery validates the full journal against input order and caller bindings;
    final predictions retain input order. Resumed time covers new work only.
    """
    if not rows or concurrency < 1 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("invalid continuous workload or concurrency")
    for row in rows:
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
            raise ValueError("continuous workload prompt hash changed")
    header = {
        "format": "gleipnir-continuous-score-v1",
        "contract": contract,
        "concurrency": concurrency,
        "endpoint": ENDPOINT,
        "identities": [[r["id"], r["prompt_sha256"], r["prompt_tokens"]] for r in rows],
    }
    saved, _ = recovery_records(checkpoint, header, rows)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        with checkpoint.open("x") as handle:
            handle.write(json.dumps(header, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    remaining = iter(i for i in range(len(rows)) if i not in saved)
    observed = saved.copy()
    queue: asyncio.Queue = asyncio.Queue(maxsize=2 * concurrency)
    receipt = {
        "resumed_rows": len(saved),
        "new_rows": 0,
        "checkpoint_io_seconds": 0.0,
        "checkpoint_queue_max": 0,
        "checkpoint_backpressure_seconds": 0.0,
        "durability": "flush and fsync each writer chunk; no request-group barrier",
        "latency_scope": "HTTP through validation, excluding slot waiting and saving",
    }
    started = time.perf_counter()
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{settings['port']}",
        trust_env=False,
        timeout=settings["timeout_seconds"],
        limits=httpx.Limits(
            max_connections=concurrency, max_keepalive_connections=concurrency
        ),
        transport=transport,
    ) as client:
        with checkpoint.open("a") as handle:

            def append(entries: list[dict]) -> float:
                before = time.perf_counter()
                handle.writelines(json.dumps(e, sort_keys=True) + "\n" for e in entries)
                handle.flush()
                os.fsync(handle.fileno())
                return time.perf_counter() - before

            async def writer() -> None:
                while True:
                    item = await queue.get()
                    if item is None:
                        return
                    entries = [item]
                    finished = False
                    while len(entries) < concurrency and not queue.empty():
                        item = queue.get_nowait()
                        if item is None:
                            finished = True
                            break
                        entries.append(item)
                    receipt["checkpoint_io_seconds"] += await asyncio.to_thread(
                        append, entries
                    )
                    receipt["new_rows"] += len(entries)
                    if progress is not None:
                        progress(len(saved) + receipt["new_rows"], len(rows))
                    if finished:
                        return

            async def worker() -> None:
                for index in remaining:
                    row = rows[index]
                    before = time.perf_counter()
                    response = await client.post(
                        ENDPOINT, json={"model": "monitor", "prompt": row["prompt"]}
                    )
                    response.raise_for_status()
                    payload = response.json()
                    score = validate_score_response(payload, row["prompt_tokens"])
                    completed = time.perf_counter()
                    value = {
                        "id": row["id"],
                        "prompt_sha256": row["prompt_sha256"],
                        "prompt_tokens": row["prompt_tokens"],
                        "latency_seconds": completed - before,
                        "start_offset_seconds": before - started,
                        "completion_offset_seconds": completed - started,
                        "logits": payload["logits"],
                        **score,
                    }
                    observed[index] = value
                    before_save = time.perf_counter()
                    await queue.put({"index": index, "value": value})
                    receipt["checkpoint_backpressure_seconds"] += (
                        time.perf_counter() - before_save
                    )
                    receipt["checkpoint_queue_max"] = max(
                        receipt["checkpoint_queue_max"], queue.qsize()
                    )

            writer_task = asyncio.create_task(writer())
            workers = [
                asyncio.create_task(worker())
                for _ in range(min(concurrency, len(rows) - len(saved)))
            ]
            producers = asyncio.gather(*workers)
            try:
                done, _ = await asyncio.wait(
                    [producers, writer_task], return_when=asyncio.FIRST_COMPLETED
                )
                if writer_task in done:
                    await writer_task
                    raise RuntimeError("continuous checkpoint writer stopped early")
                await producers
            finally:
                for task in workers:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
                await asyncio.gather(producers, return_exceptions=True)
                # Writer errors propagate, and successful responses are drained
                # on request failure before the caller retires the server.
                if not writer_task.done():
                    await queue.put(None)
                await writer_task
    seconds = time.perf_counter() - started
    if len(observed) != len(rows):
        raise ValueError("continuous evaluation incomplete")
    return [observed[i] for i in range(len(rows))], seconds, receipt
