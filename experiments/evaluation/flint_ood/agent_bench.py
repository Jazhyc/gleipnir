#!/usr/bin/env python3
"""Flint BF16 on the 96 agent trajectories of the serving benchmark
(`experiments/serving/serving_concurrency/run.py --agent`): closed-loop prompt
tokens/s and request latency at concurrency 64 down to 1, three passes each.

The trajectories and their compact instruction are those of
`data/b200_inference_benchmark/full.json`. Each prompt's user text is taken
out of its Qwen chat wrapper and rendered with Flint's template and the
empty-think decision boundary. Requests go over localhost HTTP to a stock
`vllm serve` completions endpoint (one constrained token with logprobs), so
latency includes HTTP, tokenization, queueing and inference, as in the Qwen
measurements. Scores are timing payloads."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from experiments.evaluation.flint_ood.interface import (
    EMPTY_THINK,
    MODEL_ID,
    REVISION,
    render_margin_prompt,
)
from gleipnir.serving.benchmark import measurement_summary

DATA = Path("data/b200_inference_benchmark")
AGENT_SOURCES = ("bash_arena", "bash_bench", "gloom", "stride")
QWEN_HEAD = "<|im_start|>user\n"
QWEN_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nPrediction:"
LEVELS = (64, 32, 16, 8, 4, 2, 1)
REPEATS = 3
PORT = 8010
SERVER_ARGS = [
    "--served-model-name", "flint",
    "--dtype", "bfloat16",
    "--max-model-len", "32768",
    "--max-num-seqs", "128",
    "--max-num-batched-tokens", "32768",
    "--no-enable-prefix-caching",
    "--block-size", "16",
    "--seed", "20260830",
    "--host", "127.0.0.1",
    "--port", str(PORT),
]  # fmt: skip


def user_text(prompt: str) -> str:
    if not (prompt.startswith(QWEN_HEAD) and prompt.endswith(QWEN_TAIL)):
        raise ValueError("benchmark prompt is not in the frozen Qwen wrapper")
    return prompt[len(QWEN_HEAD) : -len(QWEN_TAIL)]


def agent_rows(tokenizer) -> list[dict]:
    manifest = json.loads((DATA / "manifest.json").read_text())
    path = DATA / "full.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"]["full"]:
        raise ValueError("frozen workload drift")
    names = {f"tool_trajectory/{name}" for name in AGENT_SOURCES}
    rows = []
    for row in json.loads(path.read_text()):
        if row["dataset"] not in names:
            continue
        text = render_margin_prompt(
            tokenizer, user_text(row["prompt"]), assistant_prefill=EMPTY_THINK
        )
        rows.append(
            {
                "id": row["id"],
                "prompt": text,
                "prompt_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                "qwen_prompt_tokens": row["prompt_tokens"],
            }
        )
    if len(rows) != 96:
        raise ValueError("expected the 96 agent trajectories")
    return rows


async def trial(rows: list[dict], concurrency: int, token_ids: list[int]):
    import httpx

    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{PORT}",
        trust_env=False,
        timeout=1800,
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
    ) as client:

        async def one(row):
            async with semaphore:
                before = time.perf_counter()
                response = await client.post(
                    "/v1/completions",
                    json={
                        "model": "flint",
                        "prompt": row["prompt"],
                        "max_tokens": 1,
                        "temperature": 0,
                        "logprobs": 2,
                        "allowed_token_ids": token_ids,
                        "add_special_tokens": False,
                    },
                )
                response.raise_for_status()
                elapsed = time.perf_counter() - before
                body = response.json()
                if body["usage"]["prompt_tokens"] != row["prompt_tokens"]:
                    raise ValueError("endpoint token count differs")
                return {
                    "id": row["id"],
                    "prompt_tokens": row["prompt_tokens"],
                    "latency_seconds": elapsed,
                    "text": body["choices"][0]["text"],
                }

        before = time.perf_counter()
        values = await asyncio.gather(*(one(row) for row in rows))
        return values, time.perf_counter() - before


async def measure(out: Path, rows: list[dict], token_ids: list[int]) -> dict:
    await trial(rows, LEVELS[0], token_ids)  # excluded warmup pass
    summary = {}
    for concurrency in LEVELS:
        passes = []
        for repeat in range(REPEATS):
            values, seconds = await trial(rows, concurrency, token_ids)
            (out / f"full_c{concurrency}_repeat{repeat}.json").write_text(
                json.dumps({"rows": values, "seconds": seconds}) + "\n"
            )
            passes.append(measurement_summary(values, seconds))
            print(
                f"c{concurrency} repeat {repeat}: "
                f"{passes[-1]['prompt_tokens_per_second']:.0f} prompt tok/s, "
                f"p50 {1000 * passes[-1]['latency']['p50_seconds']:.1f} ms",
                flush=True,
            )
        summary[f"c{concurrency}"] = {
            "prompt_tokens_per_second": statistics.median(
                p["prompt_tokens_per_second"] for p in passes
            ),
            "requests_per_second": statistics.median(
                p["requests_per_second"] for p in passes
            ),
            "p50_ms": 1000
            * statistics.median(p["latency"]["p50_seconds"] for p in passes),
            "p95_ms": 1000
            * statistics.median(p["latency"]["p95_seconds"] for p in passes),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    import httpx
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    token_ids = [tokenizer.encode(v, add_special_tokens=False)[0] for v in ("0", "1")]
    rows = agent_rows(tokenizer)
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL_ID,
        "--revision", REVISION, *SERVER_ARGS,
    ]  # fmt: skip
    log = (args.output / "server.log").open("w")
    started = time.time()
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        while True:
            if server.poll() is not None:
                raise RuntimeError("vLLM server exited during startup")
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5).is_success:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(5)
        startup = time.time() - started
        print(f"server ready after {startup:.0f}s", flush=True)
        summary = asyncio.run(measure(args.output, rows, token_ids))
        flint_tokens = sum(r["prompt_tokens"] for r in rows)
        qwen_tokens = sum(r["qwen_prompt_tokens"] for r in rows)
        report = {
            "model": MODEL_ID,
            "revision": REVISION,
            "command": command,
            "startup_seconds": startup,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "rows": len(rows),
            "prompt_tokens": flint_tokens,
            "median_prompt_tokens": statistics.median(r["prompt_tokens"] for r in rows),
            "qwen_prompt_tokens_same_rows": qwen_tokens,
            "levels": summary,
        }
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["levels"], indent=1), flush=True)
        print("AGENT_BENCH_DONE", flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
