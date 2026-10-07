"""Matched original/direct host wrappers on one unchanged GPU worker."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from experiments.b200_inference_benchmark.gigatoken_ab import (
    ModeController,
    paired_ratios,
)
from experiments.b200_inference_benchmark.run import DATA, ROOT, sha, trial, write
from gleipnir.inference_benchmark import (
    measurement_summary,
    paired_score_summary,
    ranking_comparison,
)


def select_direct(comparison: dict) -> bool:
    """Use the frozen throughput/latency rule; quality remains a separate guard."""
    c1, c128 = comparison["c1"], comparison["c128"]
    ratio = c128["paired_throughput"]
    return (
        ratio["median"] > 1.005
        and sum(r > 1 for r in ratio["ratios"]) >= 5
        and all(
            c1[key]["direct"] / c1[key]["original"] <= 1.02
            for key in ["latency_p50", "latency_p95"]
        )
        and c1["paired_scores"]["score"]["max_absolute_difference"] == 0
        and all(
            abs(c128["ranking"]["auroc_delta"][key]) <= 0.001
            for key in ["macro", "pooled"]
        )
    )


async def measure(name: str) -> None:
    import httpx

    out = ROOT / "results/b200_attention_gdn_serving" / name
    out.mkdir(exist_ok=False)
    (out / "executed_client.py").write_bytes(Path(__file__).read_bytes())
    serving = ROOT / "results/b200_attention_gdn_serving"
    server = json.loads((serving / "server.json").read_text())
    engine = json.loads((serving / "loaded_precision.json").read_text())["worker_pid"]
    identity = ModeController(server, engine)
    if identity.read()["mode"] != "native" or not server.get("host_wrapper"):
        raise ValueError(
            "host comparison requires the admitted native frontend/host wrapper"
        )
    manifest = json.loads((DATA / "manifest.json").read_text())
    workloads = {
        key: json.loads((DATA / f"{key}.json").read_text()) for key in ["quick", "full"]
    }
    for key in workloads:
        if sha(DATA / f"{key}.json") != manifest["files"][key]:
            raise ValueError("frozen workload drift")
    report = {
        "status": "running",
        "server": server,
        "engine_pid": engine,
        "trials": [],
        "manifest_sha256": sha(DATA / "manifest.json"),
        "baseline_sha256": sha(
            ROOT / "experiments/b200_inference_benchmark/baseline.json"
        ),
        "gpu_compile_identity": json.loads(
            (serving / "compile_identity.json").read_text()
        )["hash"],
        "source_sha256": sha(Path(__file__)),
        "selected": "original",
    }

    async def state(mode=None):
        if identity.read()["mode"] != "native":
            raise ValueError("frontend drift during wrapper comparison")
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8010", trust_env=False, timeout=180
        ) as client:
            response = await client.post(
                "/collective_rpc",
                json={
                    "method": "frost_wrapper_state",
                    "kwargs": {} if mode is None else {"mode": mode},
                    "timeout": 180,
                },
            )
            response.raise_for_status()
            results = response.json()["results"]
        if len(results) != 1 or results[0]["worker_pid"] != engine:
            raise ValueError("host control changed GPU identity")
        if mode is not None and results[0]["mode"] != mode:
            raise ValueError("host control did not select the requested mode")
        return results[0]

    async def infer(rows, concurrency):
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8010",
            trust_env=False,
            timeout=300,
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
        ) as client:
            return await trial(client, rows, manifest["token_ids"], concurrency)

    results = {}
    completed = False
    try:
        write(out / "summary.json", report)
        for concurrency, key, repeats in [(1, "quick", 3), (128, "full", 6)]:
            rows = workloads[key]
            results[concurrency] = {"original": [], "direct": []}
            for mode in results[concurrency]:
                await state(mode)
                values, seconds = await infer(rows, concurrency)
                write(
                    out / f"warmup_c{concurrency}_{mode}.json",
                    {"seconds": seconds, "values": values},
                )
                print("wrapper_warmup", concurrency, mode, flush=True)
            for pair in range(repeats):
                for mode in (
                    ["original", "direct"] if pair % 2 == 0 else ["direct", "original"]
                ):
                    before = await state(mode)
                    values, seconds = await infer(rows, concurrency)
                    after = await state()
                    other = "direct" if mode == "original" else "original"
                    if (
                        after["mode"] != mode
                        or after["calls"][mode] <= before["calls"][mode]
                        or after["calls"][other] != before["calls"][other]
                    ):
                        raise ValueError("live host-wrapper callback drift")
                    results[concurrency][mode].append(values)
                    write(out / f"c{concurrency}_{mode}_pair{pair}.json", values)
                    report["trials"].append(
                        {
                            "concurrency": concurrency,
                            "pair": pair,
                            "mode": mode,
                            **measurement_summary(values, seconds),
                            "before": before,
                            "after": after,
                        }
                    )
                    write(out / "summary.json", report)
                    print(
                        "wrapper_pass",
                        concurrency,
                        pair,
                        mode,
                        round(seconds, 3),
                        flush=True,
                    )
        comparison = {}
        for concurrency, key in [(1, "quick"), (128, "full")]:
            passes = {
                mode: [
                    t
                    for t in report["trials"]
                    if t["concurrency"] == concurrency and t["mode"] == mode
                ]
                for mode in ["original", "direct"]
            }
            comparison[f"c{concurrency}"] = {
                "tokens_s": {
                    mode: statistics.median(t["prompt_tokens_per_second"] for t in ts)
                    for mode, ts in passes.items()
                },
                "latency_p50": {
                    mode: statistics.median(t["latency"]["p50_seconds"] for t in ts)
                    for mode, ts in passes.items()
                },
                "latency_p95": {
                    mode: statistics.median(t["latency"]["p95_seconds"] for t in ts)
                    for mode, ts in passes.items()
                },
                "paired_throughput": paired_ratios(
                    [t["prompt_tokens_per_second"] for t in passes["original"]],
                    [t["prompt_tokens_per_second"] for t in passes["direct"]],
                ),
                "paired_scores": paired_score_summary(
                    results[concurrency]["original"], results[concurrency]["direct"]
                ),
                "ranking": ranking_comparison(
                    workloads[key],
                    results[concurrency]["original"],
                    results[concurrency]["direct"],
                ),
            }
        report["comparison"] = comparison
        report["selected"] = "direct" if select_direct(comparison) else "original"
        # Separate paired traces; never use profiler timing for speed claims.
        profiles = serving / "profiles"
        for index, mode in enumerate(["original", "direct"]):
            await state(mode)
            existing = {p.name for p in profiles.glob("*.gz")}
            async with httpx.AsyncClient(
                base_url="http://127.0.0.1:8010", trust_env=False, timeout=600
            ) as client:
                (await client.post("/start_profile")).raise_for_status()
                try:
                    values, seconds = await infer(workloads["full"], 128)
                finally:
                    (await client.post("/stop_profile")).raise_for_status()
            new = [p for p in profiles.glob("*.gz") if p.name not in existing]
            if len(new) != 1:
                raise ValueError("unexpected profile export count")
            target = out / f"profile_{index}_{mode}"
            target.mkdir()
            (target / new[0].name).write_bytes(new[0].read_bytes())
            write(target / "predictions.json", values)
            write(
                target / "pass.json",
                {"mode": mode, "seconds": seconds, "timed_benchmark": False},
            )
            print("wrapper_profile_complete", mode, flush=True)
        report["status"] = "complete"
        completed = True
    finally:
        report["retained_state"] = await state(
            report["selected"] if completed else "original"
        )
        write(out / "summary.json", report)
    print("wrapper_complete", report["selected"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    out = ROOT / "results/b200_attention_gdn_serving" / args.name
    if Path(args.name).name != args.name or out.exists():
        raise ValueError("use a new run name")
    try:
        asyncio.run(measure(args.name))
    except BaseException as error:
        write(out / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
