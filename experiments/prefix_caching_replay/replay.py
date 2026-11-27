"""Time per-action monitoring of the 96 benchmark trajectories on one server.

For each trajectory, in the benchmark's frozen order, send one check per
tool-calling assistant turn (``prefix_caching_tokens.compute.action_ends``),
one at a time, then the whole prompt as an isolated check. Records latency,
prompt tokens and score per request, and the server's prefix-cache counters
per trajectory (read before and after its checks, not between them).
Resumable: a trajectory already in the output is skipped.

Usage (on the Pod, with a released-image server on port 8000):
``python -m experiments.prefix_caching_replay.replay --model 4b --mode cached``
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path

import httpx

from experiments.prefix_caching_tokens.compute import CLOSE, OPEN, action_ends

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()
)
DATA = ROOT / "data/b200_inference_benchmark"
OUT = ROOT / "results/prefix_caching_replay"
SOURCES = {
    f"tool_trajectory/{s}" for s in ("bash_arena", "bash_bench", "stride", "gloom")
}
METRIC = re.compile(r"^(vllm:prefix_cache_(?:queries|hits))_total\{[^}]*\}\s+(\S+)$")
WARMUP = 3


def trajectories() -> list[dict]:
    """The 96 agent trajectories of the frozen benchmark, in its order."""
    manifest = json.loads((DATA / "manifest.json").read_text())
    path = DATA / "full.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest["files"]["full"]:
        raise ValueError("frozen workload drift")
    rows = [r for r in json.loads(path.read_text()) if r["dataset"] in SOURCES]
    if len(rows) != 96:
        raise ValueError(f"expected 96 agent trajectories, got {len(rows)}")
    return rows


def checks(prompt: str) -> list[str]:
    """Check prompts before each action: prefix through the action plus suffix."""
    start = prompt.index(OPEN) + len(OPEN)
    end = prompt.index(CLOSE)
    suffix = prompt[end:]
    return [prompt[: start + e] + suffix for e in action_ends(prompt[start:end])]


async def counters(client: httpx.AsyncClient) -> dict[str, float]:
    response = await client.get("/metrics")
    response.raise_for_status()
    totals = {"vllm:prefix_cache_queries": 0.0, "vllm:prefix_cache_hits": 0.0}
    for line in response.text.splitlines():
        match = METRIC.match(line)
        if match:
            totals[match.group(1)] += float(match.group(2))
    return totals


async def score(client: httpx.AsyncClient, prompt: str) -> dict:
    started = time.perf_counter()
    response = await client.post(
        "/v1/monitor/score", json={"model": "monitor", "prompt": prompt}
    )
    response.raise_for_status()
    latency = time.perf_counter() - started
    body = response.json()
    if not all(abs(body[k]) < float("inf") for k in ("score", "margin")):
        raise ValueError("nonfinite score")
    return {
        "prompt_tokens": body["prompt_tokens"],
        "latency_seconds": latency,
        "score": body["score"],
        "margin": body["margin"],
    }


async def reset(client: httpx.AsyncClient) -> bool:
    response = await client.post("/reset_prefix_cache")
    return response.status_code < 400


async def main_async(model: str, mode: str, name: str) -> None:
    out = OUT / name / f"{model}_{mode}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text().splitlines()}
    rows = trajectories()
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8000", trust_env=False, timeout=600
    ) as client:
        for prompt in checks(rows[0]["prompt"])[:WARMUP]:
            await score(client, prompt)  # untimed warmup
        with out.open("a") as handle:
            for row in rows:
                if row["id"] in done:
                    continue
                was_reset = await reset(client) if mode == "cached" else None
                before = await counters(client)
                values = [await score(client, p) for p in checks(row["prompt"])]
                after = await counters(client)
                isolated = await score(client, row["prompt"])
                handle.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "dataset": row["dataset"],
                            "label": row["label"],
                            "cache_reset": was_reset,
                            # Over the trajectory's checks only: scraping
                            # /metrics between checks may add latency.
                            "cache_queried_tokens": after["vllm:prefix_cache_queries"]
                            - before["vllm:prefix_cache_queries"],
                            "cache_hit_tokens": after["vllm:prefix_cache_hits"]
                            - before["vllm:prefix_cache_hits"],
                            "checks": values,
                            "isolated": isolated,
                        }
                    )
                    + "\n"
                )
                handle.flush()
                seconds = sum(v["latency_seconds"] for v in values)
                print(
                    f"{model} {mode} {row['id']} checks {len(values)} "
                    f"{seconds:.2f}s isolated {isolated['latency_seconds']:.3f}s",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("4b", "27b"), required=True)
    parser.add_argument("--mode", choices=("cached", "uncached"), required=True)
    parser.add_argument("--name", default="replay01")
    args = parser.parse_args()
    asyncio.run(main_async(args.model, args.mode, args.name))


if __name__ == "__main__":
    main()
