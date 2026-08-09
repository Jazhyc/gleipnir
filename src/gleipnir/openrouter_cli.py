"""CLI for resumable OpenRouter binary-logprob annotation."""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from gleipnir.openrouter import OpenRouterConfig, PromptRecord, score_prompt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default=OpenRouterConfig.endpoint)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument(
        "--reasoning-effort",
        choices=("max", "xhigh", "high", "medium", "low", "minimal", "none"),
        default="none",
    )
    parser.add_argument(
        "--provider-sort",
        choices=("price", "throughput", "latency"),
        default="price",
    )
    parser.add_argument("--provider-only")
    parser.add_argument(
        "--allow-fallbacks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def load_prompts(path: Path, limit: int | None = None) -> list[PromptRecord]:
    """Load unique JSONL rows with stable ``id`` and ``prompt`` fields."""
    records = []
    seen = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row or "prompt" not in row:
                raise ValueError(f"{path}:{line_number} requires id and prompt")
            record_id = str(row["id"])
            if record_id in seen:
                raise ValueError(f"duplicate input id {record_id!r}")
            seen.add(record_id)
            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"metadata must be an object for id={record_id!r}")
            records.append(
                PromptRecord(
                    record_id=record_id,
                    prompt=str(row["prompt"]),
                    metadata=metadata,
                )
            )
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"no prompt records found in {path}")
    return records


def load_cache(
    path: Path,
    work: dict[str, PromptRecord],
) -> dict[str, dict[str, Any]]:
    """Load matching cached rows and reject prompt-identity drift."""
    cached: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cached
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = str(row["id"])
            item = work.get(record_id)
            if item is None:
                continue
            if row.get("prompt_sha256") != item.prompt_sha256:
                raise ValueError(
                    f"cached prompt mismatch at {path}:{line_number} "
                    f"for id={record_id!r}"
                )
            cached[record_id] = row
    return cached


def atomic_write(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    """Replace a cache atomically in stable record-id order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record_id in sorted(rows):
            handle.write(json.dumps(rows[record_id], sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing")

    records = load_prompts(args.input, args.limit)
    work = {record.record_id: record for record in records}
    cached = load_cache(args.output, work)
    pending = [record for record in records if record.record_id not in cached]
    config = OpenRouterConfig(
        model=args.model,
        endpoint=args.endpoint,
        max_tokens=args.max_tokens,
        top_logprobs=args.top_logprobs,
        reasoning_effort=args.reasoning_effort,
        provider_sort=args.provider_sort,
        provider_only=args.provider_only,
        allow_fallbacks=args.allow_fallbacks,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
    )
    print(
        f"prepared={len(records)} cached={len(cached)} pending={len(pending)} "
        f"model={config.model}",
        flush=True,
    )

    lock = threading.Lock()
    failures = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(score_prompt, record, config, api_key): record
            for record in pending
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                result = future.result()
            except Exception as error:
                failures.append(f"{record.record_id}: {error}")
                print(f"FAILED {failures[-1]}", flush=True)
                continue
            with lock:
                cached[record.record_id] = result
                atomic_write(args.output, cached)
            print(f"completed={len(cached)}/{len(records)}", flush=True)

    atomic_write(args.output, cached)
    if failures or len(cached) != len(records):
        print(f"incomplete failures={len(failures)}; rerun to resume", flush=True)
        return 1
    print(f"complete rows={len(cached)} output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
