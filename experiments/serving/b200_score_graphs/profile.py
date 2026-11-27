"""Confirm actual graph replay with separate bypass/replay traces, after timing."""

import argparse
import asyncio
import gzip
import json
from collections import Counter
from pathlib import Path

import httpx

from experiments.b200_attention_gdn_serving.prefill_graph_canary import graph_state
from experiments.b200_inference_benchmark.run import DATA, ROOT, sha, write
from experiments.b200_monitor_score.run import SERVING, trial


def summarize(path):
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("dur", 0) > 0]
    if not kernels:
        raise ValueError("profile contains no CUDA kernel events")
    names = Counter(
        e["name"] for e in events if e.get("cat") in {"cuda_runtime", "cuda_driver"}
    )
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    busy = sum(b - a for a, b in merged)
    window = merged[-1][1] - merged[0][0]
    return dict(
        trace_sha256=sha(path),
        kernels=len(kernels),
        kernel_duration_sum_us=sum(e["dur"] for e in kernels),
        gpu_window_us=window,
        gpu_busy_union_us=busy,
        gpu_gaps_us=window - busy,
        graph_launches=names["cudaGraphLaunch"],
        ordinary_launches=sum(
            v
            for n, v in names.items()
            if n.startswith(("cudaLaunchKernel", "cuLaunchKernel"))
        ),
        launch_api_counts={n: v for n, v in names.items() if "Launch" in n},
        diagnostic_only=True,
    )


async def profile(name):
    out = ROOT / "results/b200_score_graphs" / name
    if json.loads((out / "summary.json").read_text())["status"] != "complete":
        raise ValueError("finish unprofiled benchmark before profiling")
    settings = json.loads((out / "settings.json").read_text())
    rows = json.loads((DATA / "full.json").read_text())
    profiles = SERVING / "profiles"
    report = {}
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8010", trust_env=False, timeout=600
    ) as client:
        try:
            for mode, enabled in [("bypass", "false"), ("replay", "true")]:
                before = await graph_state(client, enabled)
                # Exclude toggled-mode warmup from the trace, too.
                await trial(rows, 128, settings)
                existing = {p.name for p in profiles.glob("*.gz")}
                (await client.post("/start_profile")).raise_for_status()
                try:
                    values, seconds = await trial(rows, 128, settings)
                finally:
                    (await client.post("/stop_profile")).raise_for_status()
                new = [p for p in profiles.glob("*.gz") if p.name not in existing]
                if len(new) != 1:
                    raise ValueError(f"unexpected profile exports: {len(new)}")
                target = out / f"profile_{mode}"
                target.mkdir(exist_ok=False)
                trace = target / new[0].name
                trace.write_bytes(new[0].read_bytes())
                write(target / "predictions.json", values)
                state = await graph_state(client)
                report[mode] = {
                    **summarize(trace),
                    "seconds_instrumented": seconds,
                    "dispatch_before": before,
                    "dispatch_after": state,
                }
                write(target / "summary.json", report[mode])
                print("graph_profile", mode, report[mode]["graph_launches"], flush=True)
        finally:
            report["retained_state"] = await graph_state(client, "true")
            write(out / "profiles.json", report)
    if report["bypass"]["graph_launches"] or report["replay"]["graph_launches"] <= 0:
        raise ValueError("profile does not confirm bypass/replay")
    print("graph_replay_verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("invalid run stem")
    asyncio.run(profile(args.name))
