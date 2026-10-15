"""Do short-prompt scores depend on batch composition (FP4 SwiGLU row policy)?

Score short canary prompts alone (C1) and concurrently with a filler prompt
that pushes the scheduled step past 4,096 tokens. Each condition repeats; if
scores repeat exactly within a condition but differ between conditions, the
"noise" is deterministic batch-dependent arithmetic, not nondeterminism.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys

import httpx

from experiments.b200_attention_precision.startup import startup
from experiments.b200_host_overhead.probe import diagnostic_command
from experiments.b200_inference_benchmark.run import DATA, ROOT, write
from experiments.b200_long_context.run import gpu, rpc

URL = "http://127.0.0.1:8010/v1/monitor/score"
WORKLOAD = ROOT / "results/b200_context_million/million01/workload.json"


async def score(client: httpx.AsyncClient, prompt: str) -> float:
    response = await client.post(URL, json={"model": "monitor", "prompt": prompt})
    response.raise_for_status()
    return response.json()["score"]


async def run(name: str, repeats: int) -> None:
    out = ROOT / "results/b200_swiglu_batching" / name
    out.mkdir(parents=True, exist_ok=False)
    startup_name = f"{name}_server"
    # The timing worker records scheduled tokens per engine step.
    await startup(startup_name, transform=diagnostic_command)
    try:
        canary = json.loads((DATA / "canary.json").read_text())
        short = [r for r in canary if r["prompt_tokens"] < 1000][:4]
        workload = {r["prompt_tokens"]: r for r in json.loads(WORKLOAD.read_text())}
        blocker, big, small = workload[32768], workload[4096], workload[1024]
        report = []
        async with httpx.AsyncClient(trust_env=False, timeout=600) as client:

            async def queued(filler: dict, row: dict) -> tuple[float, list[int]]:
                """Hold the engine with a 32K step; queue filler + row behind it."""
                await rpc("host_timing", {"reset": True})
                hold = asyncio.create_task(score(client, blocker["prompt"]))
                await asyncio.sleep(0.05)
                values = await asyncio.gather(
                    score(client, filler["prompt"]), score(client, row["prompt"])
                )
                await hold
                steps = (await rpc("host_timing", {"reset": True}))["steps"]
                return values[1], [s["tokens"] for s in steps]

            for row in short:
                alone = [await score(client, row["prompt"]) for _ in range(repeats)]
                entry = {
                    "id": row["id"],
                    "prompt_tokens": row["prompt_tokens"],
                    "alone": alone,
                }
                for label, filler in (
                    ("with_4k_filler", big),
                    ("with_1k_filler", small),
                ):
                    scores, steps = [], []
                    for _ in range(repeats):
                        value, tokens = await queued(filler, row)
                        scores.append(value)
                        steps.append(tokens)
                    shared = filler["prompt_tokens"] + row["prompt_tokens"]
                    entry[label] = {
                        "scores": scores,
                        "step_tokens": steps,
                        "co_scheduled_every_repeat": all(shared in t for t in steps),
                        "distinct": len(set(scores)),
                        "max_vs_alone": max(abs(a - b) for a in alone for b in scores),
                    }
                entry["alone_repeatable"] = len(set(alone)) == 1
                report.append(entry)
                print("swiglu_batching", json.dumps(entry), flush=True)
        write(out / "summary.json", {"rows": report})
    finally:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.b200_vllm031.stop",
                "--archive-name",
                startup_name,
            ],
            cwd=ROOT,
            check=True,
        )
        if gpu()["apps"]:
            raise ValueError("GPU not empty after verified server retirement")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run(args.name, args.repeats))


if __name__ == "__main__":
    main()
