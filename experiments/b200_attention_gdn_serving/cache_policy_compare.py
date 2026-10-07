"""Same-host resident/fresh cache-policy screen using the deployed recipes."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import httpx
import yaml

from experiments.b200_attention_gdn_serving.run import OUTPUT, resolve_condition
from experiments.b200_inference_benchmark.run import (
    DATA,
    EXPERIMENT,
    ROOT,
    benchmark,
    prepared_manifest,
    sha,
    trial,
    verify_merged_model,
    write,
)

# Historical import aliases also work on the unchanged deployed source snapshot.
from gleipnir.inference_benchmark import (
    measurement_summary,
    paired_score_summary,
    ranking_comparison,
)

AUDITS = [
    "server.json",
    "loaded_precision.json",
    "compile_identity.json",
    "native_attention.json",
    "native_preparation.json",
    "native_gemm_tuning.json",
    "native_attention_projections.json",
    "native_swiglu_output.json",
]


def process_stats(pid: int) -> dict:
    """Read process CPU execution and scheduler waits without profiling."""
    directory = Path(f"/proc/{pid}")
    stat = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
    ticks = os.sysconf("SC_CLK_TCK")
    scheduler = (directory / "schedstat").read_text().split()
    return {
        "pid": pid,
        "cpu_seconds": (int(stat[11]) + int(stat[12])) / ticks,
        "scheduler_run_seconds": int(scheduler[0]) / 1e9,
        "scheduler_wait_seconds": int(scheduler[1]) / 1e9,
        "affinity": sorted(os.sched_getaffinity(pid)),
    }


def snapshot() -> dict:
    """Record live identity and counters separately from HTTP timing."""
    server = json.loads((OUTPUT / "server.json").read_text())
    worker = json.loads((OUTPUT / "loaded_precision.json").read_text())["worker_pid"]
    return {
        "api": process_stats(server["pid"]),
        "engine": process_stats(worker),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,driver_version,memory.used",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        "cpu_stat": Path("/sys/fs/cgroup/cpu.stat").read_text().strip(),
    }


def comparison(rows: list[dict], baseline: dict, candidate: dict) -> dict:
    """Compare all repeats while retaining quality and latency distributions."""
    result = {}
    for concurrency in [1, 128]:
        left = [x for x in baseline["timing"] if x["concurrency"] == concurrency]
        right = [x for x in candidate["timing"] if x["concurrency"] == concurrency]
        values = {}
        for name, repeats in [("baseline", left), ("candidate", right)]:
            values[name] = {
                "input_tokens_per_second": statistics.median(
                    x["prompt_tokens_per_second"] for x in repeats
                ),
                "latency_p50_seconds": statistics.median(
                    x["latency"]["p50_seconds"] for x in repeats
                ),
                "latency_p95_seconds": statistics.median(
                    x["latency"]["p95_seconds"] for x in repeats
                ),
                "throughput_range": [
                    min(x["prompt_tokens_per_second"] for x in repeats),
                    max(x["prompt_tokens_per_second"] for x in repeats),
                ],
            }
        result[f"c{concurrency}"] = {
            **values,
            "throughput_change_percent": 100
            * (
                values["candidate"]["input_tokens_per_second"]
                / values["baseline"]["input_tokens_per_second"]
                - 1
            ),
            "median_latency_change_percent": 100
            * (
                values["candidate"]["latency_p50_seconds"]
                / values["baseline"]["latency_p50_seconds"]
                - 1
            ),
            "scores": paired_score_summary(
                baseline["values"][str(concurrency)],
                candidate["values"][str(concurrency)],
            ),
        }
    result["ranking"] = ranking_comparison(
        rows, baseline["values"]["128"], candidate["values"]["128"]
    )
    return result


async def infer(
    rows: list[dict], ids: list[int], concurrency: int, settings: dict
) -> tuple[list[dict], float]:
    """Use a fresh pool per pass, matching the archived wrapper benchmark."""
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{settings['port']}",
        trust_env=False,
        timeout=settings["timeout_seconds"],
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
    ) as client:
        return await trial(client, rows, ids, concurrency)


async def measure(phase: str, output: Path, manifest: dict, settings: dict) -> dict:
    output.mkdir(exist_ok=False)
    report = {
        "status": "running",
        "phase": phase,
        "timing": [],
        "values": {"1": [], "128": []},
    }
    server = json.loads((OUTPUT / "server.json").read_text())
    expected = (
        "prompt_only_worker"
        if phase != "fresh_cached"
        else "swiglu_native_output_worker"
    )
    if expected not in server["command"][server["command"].index("--worker-cls") + 1]:
        raise ValueError("wrong live worker for the requested phase")
    if server.get("status") != "ready":
        raise ValueError("server is not ready")
    report["before"] = snapshot()
    for name in AUDITS:
        if (OUTPUT / name).exists():
            write(output / name, json.loads((OUTPUT / name).read_text()))
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{settings['port']}",
        trust_env=False,
        timeout=settings["timeout_seconds"],
        limits=httpx.Limits(max_connections=256),
    ) as client:
        response = await client.post(
            "/collective_rpc",
            json={"method": "frost_wrapper_state", "kwargs": {}, "timeout": 60},
        )
        response.raise_for_status()
        state = response.json()["results"][0]
        if (
            state["mode"] != "direct"
            or state["worker_pid"] != report["before"]["engine"]["pid"]
        ):
            raise ValueError("direct FROST host identity drift")
        if phase != "fresh_cached":
            response = await client.post(
                "/collective_rpc",
                json={"method": "prompt_only_state", "kwargs": {}, "timeout": 60},
            )
            response.raise_for_status()
            report["cache_state"] = response.json()["results"][0]
            if any(
                report["cache_state"][key]
                for key in ["runner_cache_bytes", "layer_cache_bytes", "cache_specs"]
            ):
                raise ValueError("cache-free phase retains persistent caches")
        if phase == "resident_prompt_only":
            response = await client.post(
                "/collective_rpc",
                json={"method": "prompt_only_trace_state", "kwargs": {}, "timeout": 60},
            )
            response.raise_for_status()
            if response.json()["results"][0]["mode"] != "off":
                raise ValueError("resident operator tracing is still active")
        for concurrency, key, repeats in [
            (1, "quick", settings["c1_repeats"]),
            (128, "full", settings["c128_repeats"]),
        ]:
            rows = json.loads((DATA / f"{key}.json").read_text())
            if sha(DATA / f"{key}.json") != manifest["files"][key]:
                raise ValueError("frozen workload drift")
            for index in range(settings["warmup_repeats_per_concurrency"]):
                values, elapsed = await infer(
                    rows, manifest["token_ids"], concurrency, settings
                )
                write(
                    output / f"warmup_c{concurrency}_{index}.json",
                    {"values": values, "seconds": elapsed, "excluded": True},
                )
            for repeat in range(repeats):
                before = {
                    name: process_stats(report["before"][name]["pid"])
                    for name in ["api", "engine"]
                }
                values, elapsed = await infer(
                    rows, manifest["token_ids"], concurrency, settings
                )
                after = {name: process_stats(before[name]["pid"]) for name in before}
                value = {
                    "concurrency": concurrency,
                    "repeat": repeat,
                    **measurement_summary(values, elapsed),
                    "process_counter_deltas": {
                        name: {
                            key: after[name][key] - before[name][key]
                            for key in [
                                "cpu_seconds",
                                "scheduler_run_seconds",
                                "scheduler_wait_seconds",
                            ]
                        }
                        for name in before
                    },
                }
                report["timing"].append(value)
                report["values"][str(concurrency)].append(values)
                write(output / f"c{concurrency}_repeat{repeat}.json", values)
                write(
                    output / "summary.json",
                    {k: v for k, v in report.items() if k != "values"},
                )
                print(
                    "pass_complete",
                    phase,
                    concurrency,
                    repeat,
                    round(elapsed, 3),
                    flush=True,
                )
    report["after"] = snapshot()
    for name in ["api", "engine"]:
        if report["after"][name]["pid"] != report["before"][name]["pid"]:
            raise ValueError("worker changed during measurement")
    report["status"] = "complete"
    write(output / "summary.json", {k: v for k, v in report.items() if k != "values"})
    return report


async def launch(phase: str, output: Path, base: dict, manifest: dict) -> None:
    os.environ.pop("GLEIPNIR_PROMPT_ONLY_TRACE", None)
    os.environ.pop("GLEIPNIR_PROMPT_ONLY_TRACE_MODE", None)
    recipe = (
        "prompt_only.json"
        if phase == "fresh_prompt_only"
        else "fp4_swiglu_native_output.json"
    )
    raw = json.loads(Path(__file__).with_name(recipe).read_text())
    raw["name"] = output.name
    selection = json.loads((EXPERIMENT / "baseline.json").read_text())
    previous = json.loads((ROOT / selection["results"] / "condition.json").read_text())
    argv = previous["extra_server_args"]
    sources = set(
        json.loads(argv[argv.index("--additional-config") + 1])["gleipnir_frost_fp4"]
    )
    if phase == "fresh_prompt_only":
        sources.update(
            [
                "src/gleipnir/serving_prompt_only.py",
                "src/gleipnir/serving_prompt_only_contract.py",
                "experiments/b200_attention_gdn_serving/prompt_only_worker.py",
                "experiments/b200_attention_gdn_serving/prompt_only_canary.py",
            ]
        )
    condition = resolve_condition(raw, {p: sha(ROOT / p) for p in sorted(sources)})
    config = {**base, **raw["serving_config_overrides"], "port": raw["port"]}
    output.mkdir(exist_ok=False)
    write(output / "condition.json", condition)
    write(output / "manifest.json", manifest)
    for relative in [
        *sources,
        str(Path(__file__).relative_to(ROOT)),
        "experiments/b200_inference_benchmark/run.py",
    ]:
        path = output / "executed_sources" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    merged = Path(raw["merged_model"])
    write(output / "merged_artifact.json", verify_merged_model(base, merged))
    await benchmark(
        config, manifest, output, 64, False, merged, condition, startup_only=True
    )
    parity = json.loads((output / "http_parity.json").read_text())
    if not (parity.get("passed") or parity.get("evaluation_passed")):
        raise ValueError("startup scoring parity failed")


async def run(name: str) -> None:
    out = OUTPUT / name
    out.mkdir(exist_ok=False)
    settings = json.loads(Path(__file__).with_suffix(".json").read_text())
    write(out / "settings.json", settings)
    (out / "executed_client.py").write_bytes(Path(__file__).read_bytes())
    base = yaml.safe_load((EXPERIMENT / "config.yaml").read_text())
    manifest = prepared_manifest(base)
    write(out / "manifest.json", manifest)
    results = {}
    report = {
        "status": "running",
        "same_host": True,
        "promotion": False,
        "phases_completed": [],
    }
    write(out / "summary.json", report)
    try:
        for phase in settings["phases"]:
            if phase != "resident_prompt_only":
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "experiments.b200_attention_gdn_serving.stop_server",
                        "--archive-name",
                        name + "_before_" + phase,
                        "--reason",
                        "Authorized same-host cache-policy comparison",
                    ],
                    cwd=ROOT,
                    check=True,
                )
                await launch(
                    phase, OUTPUT / (name + "_" + phase + "_start"), base, manifest
                )
            results[phase] = await measure(phase, out / phase, manifest, settings)
            identity = results[phase]["before"]["gpu"].split(",")[:2]
            if phase == "resident_prompt_only":
                report["gpu_identity"] = identity
            elif identity != report["gpu_identity"]:
                raise ValueError("GPU or driver changed during the comparison")
            report["phases_completed"].append(phase)
            write(out / "summary.json", report)
        full = json.loads((DATA / "full.json").read_text())
        report["cache_policy"] = comparison(
            full, results["fresh_cached"], results["fresh_prompt_only"]
        )
        report["restart"] = comparison(
            full, results["resident_prompt_only"], results["fresh_prompt_only"]
        )
        baseline = (
            ROOT / json.loads((EXPERIMENT / "baseline.json").read_text())["results"]
        )
        reference = {
            "values": {
                str(c): [
                    json.loads(p.read_text()) for p in sorted(baseline.glob(pattern))
                ]
                for c, pattern in [
                    (1, "c1_repeat*.json"),
                    (128, "high_c128_repeat*.json"),
                ]
            }
        }
        report["archived_reference_scores"] = {
            phase: {
                "c1": paired_score_summary(
                    reference["values"]["1"], result["values"]["1"]
                ),
                "c128": paired_score_summary(
                    reference["values"]["128"], result["values"]["128"]
                ),
                "ranking": ranking_comparison(
                    full, reference["values"]["128"], result["values"]["128"]
                ),
            }
            for phase, result in results.items()
        }
        report["status"] = "complete"
        write(out / "summary.json", report)
        print(
            "comparison_complete", json.dumps(report["cache_policy"]["c1"]), flush=True
        )
    except BaseException as error:
        write(out / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        report["status"] = "failed"
        write(out / "summary.json", report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        parser.error("name must be a new directory stem")
    asyncio.run(run(args.name))


if __name__ == "__main__":
    main()
