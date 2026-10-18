"""Paired in-server A/B of GDN prefill routes for the BF16 27B at C1.

Same arms and synthetic lengths as ``experiments/b200_gdn_route/paired.py``
(``auto`` lets FlashInfer pick context parallelism for one request;
``non_cp`` forces the non-CP route for every chunk), toggled by RPC inside
one BF16 server with last-query attention and the lean GDN path, in
alternating order. Reports per-length p50 latency, paired differences and
whether scores are stable within each arm.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from experiments.b200_gdn_route.paired import ARMS, CASES, WORKLOAD
from experiments.b200_host_overhead.probe import sequential
from experiments.b200_long_context.run import rpc
from experiments.qwen38_27b_serving.bench import DATA, LOGS, MODEL, OUT, ROOT, sources
from gleipnir.data.monitoring import write_json
from gleipnir.serving import bf16_launch
from gleipnir.serving.bf16_stack import receipt_binding

WORKER = "gleipnir.serving.bf16_stack.Bf16GatedLastQueryFastGdnWorker"
RECEIPT = ROOT / "results/qwen38_27b_serving/last_query_bf16.json"


async def sweep(out: Path, rounds: int) -> dict:
    workload = {r["prompt_tokens"]: r for r in json.loads(WORKLOAD.read_text())}
    cases = {
        case: [workload[length]] * count
        for group in ("short", "crossover")
        for case, (length, count) in CASES[group].items()
    }
    cases["quick64"] = json.loads((DATA / "quick.json").read_text())
    results = {case: {arm: [] for arm in ARMS} for case in cases}
    for index in range(rounds):
        order = list(ARMS) if index % 2 == 0 else list(reversed(ARMS))
        for arm in order:
            await rpc("set_gdn_route", {"min_tokens": ARMS[arm]})
            for case, rows in cases.items():
                await sequential(rows[:3])
                values = await sequential(rows)
                results[case][arm].append(
                    {
                        "p50_ms": 1000
                        * statistics.median(v["latency_seconds"] for v in values),
                        "scores": [v["score"] for v in values],
                    }
                )
        write_json(out / "rounds.json", results)
    summary = {}
    for case, arms in results.items():
        pairs = [
            a["p50_ms"] - b["p50_ms"]
            for a, b in zip(arms["non_cp"], arms["auto"], strict=True)
        ]
        summary[case] = {
            "median_p50_ms": {
                a: statistics.median(r["p50_ms"] for r in v) for a, v in arms.items()
            },
            "paired_difference_ms": pairs,
            "non_cp_faster_rounds": sum(d < 0 for d in pairs),
            "scores_stable_within_arm": all(
                r["scores"] == arms[a][0]["scores"] for a in arms for r in arms[a]
            ),
        }
    summary["audit"] = await rpc("gdn_route_audit")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=False)
    additional = {
        **receipt_binding(ROOT, RECEIPT),
        "gdn_route": {"cp_min_tokens": ARMS["auto"]},
    }
    selection, command = bf16_launch.bf16_command(
        ROOT, MODEL, sources(), worker=WORKER, additional=additional
    )
    env = bf16_launch.environment(ROOT, selection, out / "bf16_audit.json")
    process = bf16_launch.start(
        ROOT, command, env, LOGS / f"{args.name}.log", out / "server.json"
    )
    try:
        bf16_launch.wait_ready(process, 8010)
        summary = asyncio.run(sweep(out, args.rounds))
        write_json(out / "summary.json", summary)
        print(json.dumps(summary, indent=1)[:3000], flush=True)
    finally:
        bf16_launch.stop(process)
    print(f"route_sweep_complete {args.name}", flush=True)


if __name__ == "__main__":
    main()
