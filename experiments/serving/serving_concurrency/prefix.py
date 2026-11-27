"""Compare cached and uncached scores of the 4B, one request at a time.

For every benchmark prompt: score it after a cache reset (``uncached``),
score it again (``replay``: everything but the tail is cached), then reset,
score its first half and score it once more (``half``). ``--chunk`` lowers
the prefill chunk budget, which splits an uncached prompt into several
steps as caching does, without a cache.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from experiments.serving.serving_concurrency.run import OUT, PORT, launch, prepare

CACHED = {
    "gleipnir.serving.bf16_worker.Bf16Worker": "PrefixBf16Worker",
    "experiments.b200_attention_precision.worker.AttentionPrecisionWorker": (
        "PrefixQuantWorker"
    ),
    "experiments.b200_gdn_fast.worker.FastGdnWorker": "PrefixFastWorker",
}
WORKERS = "experiments.serving.serving_concurrency.workers"


def cached_command(command: list[str]) -> list[str]:
    """Enable align-mode prefix caching and the worker that admits it."""
    command = command.copy()
    command[command.index("--no-enable-prefix-caching")] = "--enable-prefix-caching"
    worker = command.index("--worker-cls") + 1
    command[worker] = f"{WORKERS}.{CACHED[command[worker]]}"
    return [*command, "--mamba-cache-mode", "align"]


def chunked_command(command: list[str], chunk: int) -> list[str]:
    command = command.copy()
    command[command.index("--max-num-batched-tokens") + 1] = str(chunk)
    return command


async def measure(out: Path, cache: bool, limit: int | None) -> None:
    import httpx

    from experiments.b200_prefix_replay.run import cache_counters, reset
    from experiments.serving.qwen38_27b_serving.bench import workloads
    from gleipnir.data.monitoring import write_json
    from gleipnir.serving.monitor_score import ENDPOINT, validate_score_response

    rows = workloads()["full"][:limit]
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{PORT}", trust_env=False, timeout=600
    ) as client:

        async def score(prompt: str) -> dict:
            before = await cache_counters(client) if cache else {}
            started = time.perf_counter()
            response = await client.post(
                ENDPOINT, json={"model": "monitor", "prompt": prompt}
            )
            response.raise_for_status()
            latency = time.perf_counter() - started
            body = response.json()
            value = validate_score_response(body, body["prompt_tokens"])
            result = {
                "prompt_tokens": body["prompt_tokens"],
                "latency_seconds": latency,
                "score": value["score"],
                "margin": value["margin"],
            }
            if cache:
                after = await cache_counters(client)
                key = "vllm:prefix_cache_hits"
                result["hit_tokens"] = after[key] - before[key]
            return result

        records = []
        for row in rows:
            record = {"id": row["id"]}
            if cache:
                await reset(client)
            record["uncached"] = await score(row["prompt"])
            if cache:
                record["replay"] = await score(row["prompt"])
                await reset(client)
                record["first_half"] = await score(
                    row["prompt"][: len(row["prompt"]) // 2]
                )
                record["half"] = await score(row["prompt"])
            records.append(record)
        write_json(out / "prefix.json", {"rows": records})


async def evaluate(out: Path, splits: list[str]) -> None:
    """Score ID/OOD rows at the end of the trajectory after a midpoint request.

    Each row starts from an empty cache: the first half of the prompt text
    is scored (the midpoint decision, which fills the cache), then the whole
    prompt (the final decision, resumed from the cache).
    """
    import httpx

    from experiments.b200_prefix_replay.run import cache_counters, reset
    from experiments.release_serving import validate
    from experiments.release_serving.targets import TARGETS
    from gleipnir.data.monitoring import write_json, write_rows
    from gleipnir.serving.monitor_score import ENDPOINT, validate_score_response

    key = "vllm:prefix_cache_hits"
    report = {}
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{PORT}", trust_env=False, timeout=600
    ) as client:
        for split in splits:
            workload = validate.rows(TARGETS["4b"].validation[split][0])
            scored = []
            started = time.perf_counter()
            for row in workload:
                await reset(client)
                prompt = row["prompt"]
                for text in (prompt[: len(prompt) // 2], prompt):
                    before = await cache_counters(client)
                    response = await client.post(
                        ENDPOINT, json={"model": "monitor", "prompt": text}
                    )
                    response.raise_for_status()
                    body = response.json()
                    value = validate_score_response(body, body["prompt_tokens"])
                    after = await cache_counters(client)
                scored.append(
                    {
                        "id": row["id"],
                        "score": value["score"],
                        "margin": value["margin"],
                        "prompt_tokens": body["prompt_tokens"],
                        "hit_tokens": after[key] - before[key],
                    }
                )
            write_rows(out / f"{split}.jsonl", scored)
            new = {r["id"]: float(r["score"]) for r in scored}
            report[split] = {
                "rows": len(new),
                "rows_with_hit": sum(r["hit_tokens"] > 0 for r in scored),
                "pauroc_at_20": validate.macro_pauroc(
                    new, validate.labels(split), split
                ),
                "seconds": time.perf_counter() - started,
            }
            print({split: report[split]}, flush=True)
    write_json(out / "report.json", report)


def run(
    step: str,
    session: str,
    cache: bool,
    chunk: int | None,
    limit: int | None,
    splits: list[str] | None = None,
):
    from gleipnir.data.monitoring import write_json
    from gleipnir.serving import bf16_launch

    name = f"prefix_4b_{step}" + ("_cache" if cache else "")
    name += f"_chunk{chunk}" if chunk else ""
    name += "_eval" if splits else ""
    out = OUT / session / name
    out.mkdir(parents=True, exist_ok=False)
    command, env = prepare("4b", step, out)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    if cache:
        command = cached_command(command)
    if chunk:
        command = chunked_command(command, chunk)
    log = f"{session}_{name}".lstrip("_").replace("/", "_")
    process, startup, attempt = launch(command, env, log, out)
    try:
        asyncio.run(evaluate(out, splits) if splits else measure(out, cache, limit))
        write_json(
            out / "condition.json",
            {
                "step": step,
                "cache": cache,
                "chunk": chunk,
                "command": command,
                "startup_seconds": startup,
                "startup_attempt": attempt,
            },
        )
    finally:
        bf16_launch.stop(process)
    print(f"prefix_complete {name}", json.dumps({"rows": limit}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True)
    parser.add_argument("--session", default="", help="output subdirectory")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--splits", nargs="+", choices=("id", "ood"))
    args = parser.parse_args()
    if args.splits and not args.cache:
        raise ValueError("--splits scores through the cache; pass --cache")
    run(args.step, args.session, args.cache, args.chunk, args.limit, args.splits)


if __name__ == "__main__":
    main()
