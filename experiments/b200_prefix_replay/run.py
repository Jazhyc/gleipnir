"""Replay a growing monitored session with vLLM prefix caching on the full stack.

Two policies on one 1M-context YaRN server with the adopted serving workers:
check every proposed action, or a seeded ~7.2% "escalated" subset. Each policy
starts from an empty prefix cache. A final pass rescoring sampled checks after
a cache reset gives uncached latency and cached-versus-uncached score parity.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import dataclasses
import re
import time
from pathlib import Path

import httpx

from experiments.b200_context_million.run import ROOT, run
from experiments.b200_gdn_fast.run import full_stack
from experiments.b200_inference_benchmark.run import write
from experiments.b200_prefix_replay.session import load, render
from gleipnir.serving.monitor_score import validate_score_response

EXPERIMENT = Path(__file__).parent
ENDPOINT = "/v1/monitor/score"
METRIC = re.compile(r"^(vllm:prefix_cache_(?:queries|hits))_total\{[^}]*\}\s+(\S+)$")


def enable_prefix_caching(command: list[str]) -> list[str]:
    """Change only the prefix-cache policy of the long-context command."""
    command = command.copy()
    if (
        "--no-enable-prefix-caching" not in command
        or "--enable-prefix-caching" in command
    ):
        raise ValueError("expected the selected no-prefix-caching recipe")
    command[command.index("--no-enable-prefix-caching")] = "--enable-prefix-caching"
    return [*command, "--mamba-cache-mode", "align"]


async def cache_counters(client: httpx.AsyncClient) -> dict[str, float]:
    response = await client.get("/metrics")
    response.raise_for_status()
    totals = {"vllm:prefix_cache_queries": 0.0, "vllm:prefix_cache_hits": 0.0}
    for line in response.text.splitlines():
        match = METRIC.match(line)
        if match:
            totals[match.group(1)] += float(match.group(2))
    return totals


async def check(client: httpx.AsyncClient, session: dict, index: int) -> dict:
    before_counts = await cache_counters(client)
    started = time.perf_counter()
    response = await client.post(
        ENDPOINT, json={"model": "monitor", "prompt": render(session, index)}
    )
    response.raise_for_status()
    latency = time.perf_counter() - started
    after = await cache_counters(client)
    body = response.json()
    queried = (
        after["vllm:prefix_cache_queries"] - before_counts["vllm:prefix_cache_queries"]
    )
    hits = after["vllm:prefix_cache_hits"] - before_counts["vllm:prefix_cache_hits"]
    return {
        "step": index,
        "prompt_tokens": body["prompt_tokens"],
        "latency_seconds": latency,
        "cache_queried_tokens": queried,
        "cache_hit_tokens": hits,
        **validate_score_response(body, body["prompt_tokens"]),
    }


async def reset(client: httpx.AsyncClient) -> None:
    response = await client.post("/reset_prefix_cache")
    response.raise_for_status()


async def replay(rows: list[dict], settings: dict, out: Path, record: dict) -> None:
    session = load(ROOT / settings["replay_session"])
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{settings['port']}",
        trust_env=False,
        timeout=settings["timeout_seconds"],
    ) as client:
        results = {}
        for policy, steps in session["policies"].items():
            await reset(client)
            values = []
            for index in steps:
                value = await check(client, session, index)
                values.append(value)
                if len(values) % 25 == 0 or len(values) == len(steps):
                    write(out / f"replay_{policy}.json", values)
                    print(
                        "replay_progress",
                        policy,
                        len(values),
                        len(steps),
                        value["prompt_tokens"],
                        round(value["latency_seconds"], 3),
                        flush=True,
                    )
            results[policy] = values
        cumulative = session["cumulative_tokens_estimate"]
        every = session["policies"]["every_action"]
        picks = []
        for target in settings["baseline_targets"]:
            position = min(
                bisect.bisect_left([cumulative[i] for i in every], target),
                len(every) - 1,
            )
            if every[position] not in picks:
                picks.append(every[position])
        cached = {v["step"]: v for v in results["every_action"]}
        baseline = []
        for index in picks:
            await reset(client)
            value = await check(client, session, index)
            value["cached_score"] = cached[index]["score"]
            value["cached_latency_seconds"] = cached[index]["latency_seconds"]
            value["score_difference"] = value["score"] - cached[index]["score"]
            baseline.append(value)
            write(out / "baseline_uncached.json", baseline)
            print(
                "baseline",
                value["prompt_tokens"],
                round(value["latency_seconds"], 3),
                "cached",
                round(value["cached_latency_seconds"], 3),
                "score_diff",
                value["score_difference"],
                flush=True,
            )
    record["replay"] = {
        policy: {
            "checks": len(values),
            "total_latency_seconds": sum(v["latency_seconds"] for v in values),
            "hit_fraction": sum(v["cache_hit_tokens"] for v in values)
            / max(1.0, sum(v["cache_queried_tokens"] for v in values)),
            "final_prompt_tokens": values[-1]["prompt_tokens"],
        }
        for policy, values in results.items()
    }
    record["baseline_steps"] = picks


def variant():
    return dataclasses.replace(
        full_stack(),
        experiment=EXPERIMENT,
        result_group="b200_prefix_replay",
        transform=enable_prefix_caching,
        measurement=replay,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("run name must be a stem")
    asyncio.run(run(args.name, None, variant()))


if __name__ == "__main__":
    main()
