"""Repeat the c4 canary per arm in one server: numerics versus batch timing."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys

from experiments.b200_attention_precision.startup import startup
from experiments.b200_gdn_fast.paired import fast_command
from experiments.b200_inference_benchmark.run import DATA, ROOT, write
from experiments.b200_long_context.run import gpu, rpc
from experiments.b200_monitor_score.run import trial

SETTINGS = {"port": 8010, "timeout_seconds": 600}


async def run(name: str, repeats: int) -> None:
    out = ROOT / "results/b200_gdn_fast" / name
    out.mkdir(parents=True, exist_ok=False)
    startup_name = f"{name}_server"
    await startup(startup_name, transform=fast_command)
    try:
        rows = json.loads((DATA / "canary.json").read_text())
        results = {"upstream": [], "fast": [], "upstream_c1": [], "fast_c1": []}
        for _ in range(repeats):
            for arm, enabled in (("upstream", False), ("fast", True)):
                await rpc("set_gdn_fast", {"enabled": enabled})
                values, _ = await trial(rows, 4, SETTINGS)
                results[arm].append([v["score"] for v in values])
                values, _ = await trial(rows, 1, SETTINGS)
                results[f"{arm}_c1"].append([v["score"] for v in values])
        await rpc("set_gdn_fast", {"enabled": True})
        summary = {
            arm: {
                "distinct_score_vectors": len({tuple(s) for s in runs}),
                "max_spread": max(
                    max(col) - min(col) for col in zip(*runs, strict=True)
                ),
            }
            for arm, runs in results.items()
        }
        summary["c1_fast_equals_upstream"] = (
            results["fast_c1"][0] == results["upstream_c1"][0]
        )
        write(out / "runs.json", results)
        write(out / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
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
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(run(args.name, args.repeats))


if __name__ == "__main__":
    main()
