"""CLI for resumable OpenRouter binary-logprob annotation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from gleipnir.openrouter import (
    OpenRouterConfig,
    PromptRecord,
    request_settings_sha256,
    score_prompt,
)


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
        "--binary-output-mode",
        choices=("prediction_line", "scalar"),
        default="prediction_line",
    )
    parser.add_argument(
        "--structured-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Constrain scalar output to the strict JSON integer enum [0, 1].",
    )
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
    parser.add_argument(
        "--provider-order",
        action="append",
        default=[],
        help="Provider to try in order; repeat for fallbacks. Overrides provider sort.",
    )
    parser.add_argument("--provider-only")
    parser.add_argument(
        "--provider-max-prompt-price",
        type=float,
        help="Maximum accepted provider prompt price in USD per million tokens.",
    )
    parser.add_argument(
        "--provider-max-completion-price",
        type=float,
        help="Maximum accepted provider completion price in USD per million tokens.",
    )
    parser.add_argument(
        "--allow-fallbacks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sticky-routing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin the campaign to one provider endpoint with an OpenRouter session.",
    )
    parser.add_argument(
        "--session-id",
        help="Explicit OpenRouter cache/sticky session (auto-derived by default).",
    )
    parser.add_argument(
        "--warm-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Finish one pending request before concurrent fan-out.",
    )
    parser.add_argument(
        "--explicit-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mark the exact all-prompt prefix as an ephemeral provider cache block.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Atomically persist after this many new rows (default: 1).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress after this many new rows (default: 1).",
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
    settings_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load matching cached rows and reject prompt or request drift."""
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
            if (
                settings_sha256 is not None
                and row.get("request_settings_sha256") != settings_sha256
            ):
                raise ValueError(
                    f"cached request settings mismatch at {path}:{line_number} "
                    f"for id={record_id!r}"
                )
            cached[record_id] = row
    return cached


def shared_prompt_prefix(records: list[PromptRecord]) -> str:
    """Return the exact character prefix shared by all campaign prompts."""
    prefix = records[0].prompt
    for record in records[1:]:
        limit = min(len(prefix), len(record.prompt))
        index = 0
        while index < limit and prefix[index] == record.prompt[index]:
            index += 1
        prefix = prefix[:index]
        if not prefix:
            break
    return prefix


def declared_or_shared_prompt_prefix(records: list[PromptRecord]) -> str:
    """Honor a prompt-aware cache declaration, otherwise use the exact LCP."""
    declarations = {
        (
            record.metadata.get("cache_prefix_chars"),
            record.metadata.get("cache_prefix_sha256"),
        )
        for record in records
    }
    if declarations == {(None, None)}:
        return shared_prompt_prefix(records)
    if len(declarations) != 1:
        raise ValueError("prompt rows contain inconsistent cache-prefix declarations")
    prefix_chars, expected_sha256 = declarations.pop()
    if not isinstance(prefix_chars, int) or prefix_chars < 1:
        raise ValueError("cache_prefix_chars must be a positive integer on every row")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(
            "cache_prefix_sha256 must be a SHA-256 hex digest on every row"
        )
    prefix = records[0].prompt[:prefix_chars]
    actual_sha256 = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("declared cache-prefix hash does not match the first prompt")
    mismatched = [
        record.record_id for record in records if not record.prompt.startswith(prefix)
    ]
    if mismatched:
        raise ValueError(
            f"declared cache prefix is not shared by row(s) {mismatched[:3]!r}"
        )
    return prefix


def automatic_session_id(model: str, prompt_prefix: str) -> str:
    """Derive a stable, non-sensitive OpenRouter sticky-routing identifier."""
    digest = hashlib.sha256(
        (model + "\0" + prompt_prefix).encode("utf-8")
    ).hexdigest()
    return f"gleipnir-teacher-cache-{digest}"


def cache_token_totals(rows: dict[str, dict[str, Any]]) -> tuple[int, int, int]:
    """Sum prompt tokens and normalized cache reads/writes across completed rows."""
    prompt_tokens = 0
    cached_tokens = 0
    cache_write_tokens = 0
    for row in rows.values():
        usage = row.get("usage")
        if isinstance(usage, dict):
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
        cache_usage = row.get("cache_usage")
        if not isinstance(cache_usage, dict):
            continue
        cached_tokens += int(cache_usage.get("cached_tokens") or 0)
        cache_write_tokens += int(cache_usage.get("cache_write_tokens") or 0)
    return prompt_tokens, cached_tokens, cache_write_tokens


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
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be positive")
    if args.progress_every < 1:
        raise ValueError("progress-every must be positive")
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing")

    all_records = load_prompts(args.input)
    prompt_prefix = declared_or_shared_prompt_prefix(all_records)
    records = all_records if args.limit is None else all_records[: args.limit]
    work = {record.record_id: record for record in records}
    if args.session_id and not args.sticky_routing:
        raise ValueError("--session-id requires --sticky-routing")
    session_id = None
    if args.sticky_routing:
        session_key = prompt_prefix or all_records[0].prompt_sha256
        session_id = args.session_id or automatic_session_id(args.model, session_key)
    config = OpenRouterConfig(
        model=args.model,
        endpoint=args.endpoint,
        max_tokens=args.max_tokens,
        top_logprobs=args.top_logprobs,
        binary_output_mode=args.binary_output_mode,
        structured_output=args.structured_output,
        reasoning_effort=args.reasoning_effort,
        provider_sort=args.provider_sort,
        provider_order=tuple(args.provider_order),
        provider_only=args.provider_only,
        provider_max_prompt_price=args.provider_max_prompt_price,
        provider_max_completion_price=args.provider_max_completion_price,
        allow_fallbacks=args.allow_fallbacks,
        session_id=session_id,
        cache_prefix=prompt_prefix if args.explicit_cache else "",
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
    )
    settings_sha256 = request_settings_sha256(config)
    cached = load_cache(args.output, work, settings_sha256)
    pending = [record for record in records if record.record_id not in cached]
    prefix_sha256 = hashlib.sha256(prompt_prefix.encode("utf-8")).hexdigest()
    print(
        f"prepared={len(records)} cached={len(cached)} pending={len(pending)} "
        f"model={config.model} shared_prefix_chars={len(prompt_prefix)} "
        f"shared_prefix_sha256={prefix_sha256} session_id={session_id}",
        flush=True,
    )

    lock = threading.Lock()
    failures = []
    if args.warm_cache and config.cache_prefix and pending:
        warm_record = pending.pop(0)
        print(f"warming provider cache with id={warm_record.record_id}", flush=True)
        try:
            result = score_prompt(warm_record, config, api_key)
        except Exception as error:
            failures.append(f"{warm_record.record_id}: {error}")
            print(f"FAILED cache warm-up {failures[-1]}", flush=True)
        else:
            cached[warm_record.record_id] = result
            atomic_write(args.output, cached)
            print(
                f"provider cache warm-up complete id={warm_record.record_id}",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(score_prompt, record, config, api_key): record
            for record in pending
        }
        completed_since_checkpoint = 0
        completed_since_progress = 0
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
                completed_since_checkpoint += 1
                completed_since_progress += 1
                if completed_since_checkpoint >= args.checkpoint_every:
                    atomic_write(args.output, cached)
                    completed_since_checkpoint = 0
            if (
                completed_since_progress >= args.progress_every
                or len(cached) == len(records)
            ):
                print(f"completed={len(cached)}/{len(records)}", flush=True)
                completed_since_progress = 0

    atomic_write(args.output, cached)
    prompt_tokens, cached_tokens, cache_write_tokens = cache_token_totals(cached)
    cache_read_fraction = cached_tokens / prompt_tokens if prompt_tokens else 0.0
    print(
        f"provider_cache prompt_tokens={prompt_tokens} cached_tokens={cached_tokens} "
        f"cache_write_tokens={cache_write_tokens} "
        f"cache_read_fraction={cache_read_fraction:.6f}",
        flush=True,
    )
    if failures or len(cached) != len(records):
        print(f"incomplete failures={len(failures)}; rerun to resume", flush=True)
        return 1
    print(f"complete rows={len(cached)} output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
