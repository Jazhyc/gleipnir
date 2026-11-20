"""Separate per-request overhead from the replay's /metrics scrapes.

Against a running default (uncached) server, time the isolated prompts of the
first 24 benchmark trajectories one at a time, twice: plain requests, then
each request between two ``/metrics`` scrapes, as ``replay.py`` does. Compare
with the recorded c1 latency of the benchmark (48 ms median for the 4B).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx

from experiments.prefix_caching_replay.replay import OUT, counters, trajectories


async def run(model: str, name: str) -> None:
    rows = trajectories()[:24]
    out = {}
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8000", trust_env=False, timeout=600
    ) as client:
        for row in rows[:3]:
            await client.post(
                "/v1/monitor/score", json={"model": "monitor", "prompt": row["prompt"]}
            )
        for arm in ("plain", "with_metrics"):
            latencies = []
            for row in rows:
                if arm == "with_metrics":
                    await counters(client)
                started = time.perf_counter()
                response = await client.post(
                    "/v1/monitor/score",
                    json={"model": "monitor", "prompt": row["prompt"]},
                )
                response.raise_for_status()
                latencies.append(time.perf_counter() - started)
                if arm == "with_metrics":
                    await counters(client)
            out[arm] = {
                "median_ms": 1000 * statistics.median(latencies),
                "latencies": latencies,
            }
            print(arm, round(out[arm]["median_ms"], 1), flush=True)
    path = OUT / name / f"probe_{model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="4b")
    parser.add_argument("--name", default="replay01")
    args = parser.parse_args()
    asyncio.run(run(args.model, args.name))


if __name__ == "__main__":
    main()
