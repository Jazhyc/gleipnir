"""Balanced resident-worker comparisons of GPU-local CPU placement."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
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
from gleipnir.serving_cpu_placement import ProcessAffinity, parse_cpu_list


def telemetry(pids: list[int]) -> dict:
    """Record process counters and container pressure around each HTTP pass."""
    return {
        "processes": {
            str(pid): {
                "schedstat": Path(f"/proc/{pid}/schedstat").read_text().strip(),
                "stat": Path(f"/proc/{pid}/stat").read_text(),
                "status": [
                    line
                    for line in Path(f"/proc/{pid}/status").read_text().splitlines()
                    if "ctxt_switches" in line
                ],
            }
            for pid in pids
        },
        "cgroup_cpu_stat": Path("/sys/fs/cgroup/cpu.stat").read_text(),
        "cgroup_cpu_pressure": Path("/sys/fs/cgroup/cpu.pressure").read_text(),
    }


def choose_placement(comparisons: dict) -> str:
    """Prefer a consistent throughput gain without material latency regression."""
    eligible = []
    for name, comparison in comparisons.items():
        ratio = comparison["c128"]["paired_throughput"]
        latency = comparison["c1"]
        p50 = latency["latency_p50"]["candidate"] / latency["latency_p50"]["original"]
        p95 = latency["latency_p95"]["candidate"] / latency["latency_p95"]["original"]
        # Quality is a guard, not the candidate ranking/selection objective.
        quality = (
            all(
                abs(delta) <= 0.001
                for delta in comparison["c128"]["ranking"]["auroc_delta"].values()
                if isinstance(delta, (int, float))
            )
            and comparison["c1"]["paired_scores"]["score"]["max_absolute_difference"]
            == 0
        )
        speed = (
            ratio["median"] > 1.005
            and sum(v > 1 for v in ratio["ratios"]) >= 5
            and p50 <= 1.02
            and p95 <= 1.02
        ) or (p95 < 0.97 and p50 <= 1.0 and ratio["median"] >= 0.995)
        if quality and speed:
            eligible.append((ratio["median"], name))
    return max(eligible)[1] if eligible else "original"


async def measure(name: str, node: int) -> None:
    import httpx

    out = ROOT / "results/b200_inference_benchmark" / name
    out.mkdir(exist_ok=False)
    (out / "executed_client.py").write_bytes(Path(__file__).read_bytes())
    (out / "executed_affinity.py").write_bytes(
        (ROOT / "src/gleipnir/serving_cpu_placement.py").read_bytes()
    )
    serving = ROOT / "results/b200_attention_gdn_serving"
    server = json.loads((serving / "server.json").read_text())
    engine = json.loads((serving / "loaded_precision.json").read_text())["worker_pid"]
    controller = ModeController(server, engine)
    if controller.read()["mode"] != "native":
        raise ValueError(
            "CPU placement comparison requires the retained native frontend"
        )
    api, worker = ProcessAffinity(server["pid"]), ProcessAffinity(engine)
    local = parse_cpu_list(
        Path(f"/sys/devices/system/node/node{node}/cpulist").read_text()
    )
    local &= api.leader_original & worker.leader_original
    # Select four physical cores and all their SMT siblings for API processing.
    cores: dict[tuple[int, int], set[int]] = {}
    for cpu in sorted(local):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = tuple(
            int((topology / field).read_text())
            for field in ["physical_package_id", "core_id"]
        )
        cores.setdefault(key, set()).add(cpu)
    if len(cores) < 8:
        raise ValueError("GPU-local node has too few physical cores for this protocol")
    api_cpus = set().union(*list(cores.values())[:4])
    configurations = {
        "original": (None, None),
        "local": (local, local),
        "split": (api_cpus, local - api_cpus),
    }
    manifest = json.loads((DATA / "manifest.json").read_text())
    workloads = {
        key: json.loads((DATA / f"{key}.json").read_text()) for key in ["quick", "full"]
    }
    for key in workloads:
        if sha(DATA / f"{key}.json") != manifest["files"][key]:
            raise ValueError("frozen workload drift")
    report = {
        "status": "running",
        "api_pid": api.pid,
        "engine_pid": worker.pid,
        "server": server,
        "node": node,
        "configurations": {
            key: {"api": sorted(a) if a else None, "engine": sorted(b) if b else None}
            for key, (a, b) in configurations.items()
        },
        "original_affinity": {"api": api.apply(None), "engine": worker.apply(None)},
        "manifest_sha256": sha(DATA / "manifest.json"),
        "baseline_sha256": sha(Path(__file__).parent / "baseline.json"),
        "gpu_compile_identity": json.loads(
            (serving / "compile_identity.json").read_text()
        )["hash"],
        "source_sha256": sha(Path(__file__)),
        "memory_policy_changed": False,
        "client_affinity": sorted(os.sched_getaffinity(0)),
        "trials": [],
        "reused_validation": server["frontend"]["validation"],
        "selected": "original",
    }

    def placement(mode):
        if controller.read()["mode"] != "native":
            raise ValueError("frontend mode changed during placement comparison")
        a, b = configurations[mode]
        return {"api": api.apply(a), "engine": worker.apply(b)}

    async def infer(rows, concurrency):
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8010",
            timeout=300,
            trust_env=False,
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
        ) as client:
            return await trial(client, rows, manifest["token_ids"], concurrency)

    completed = False
    try:
        write(out / "summary.json", report)
        values_by_workload = {}
        for concurrency, key, repetitions in [(1, "quick", 3), (128, "full", 6)]:
            rows = workloads[key]
            values_by_workload[concurrency] = {mode: [] for mode in configurations}
            for mode in configurations:
                placement(mode)
                values, seconds = await infer(rows, concurrency)
                write(
                    out / f"warmup_c{concurrency}_{mode}.json",
                    {"seconds": seconds, "values": values},
                )
                print("placement_warmup", concurrency, mode, flush=True)
            orders = (
                list(itertools.permutations(configurations))
                if repetitions == 6
                else [
                    ("original", "local", "split"),
                    ("local", "split", "original"),
                    ("split", "original", "local"),
                ]
            )
            for pair, order in enumerate(orders):
                for mode in order:
                    masks = placement(mode)
                    before = telemetry([api.pid, worker.pid])
                    values, seconds = await infer(rows, concurrency)
                    after = telemetry([api.pid, worker.pid])
                    # Verify identity/masks after the pass, including any new threads.
                    verified = placement(mode)
                    values_by_workload[concurrency][mode].append(values)
                    write(out / f"c{concurrency}_{mode}_pair{pair}.json", values)
                    report["trials"].append(
                        {
                            "mode": mode,
                            "pair": pair,
                            "concurrency": concurrency,
                            **measurement_summary(values, seconds),
                            "affinity_before": masks,
                            "affinity_after": verified,
                            "telemetry_before": before,
                            "telemetry_after": after,
                        }
                    )
                    write(out / "summary.json", report)
                    print(
                        "placement_pass",
                        concurrency,
                        pair,
                        mode,
                        round(seconds, 3),
                        flush=True,
                    )
        comparisons = {}
        for mode in ["local", "split"]:
            comparison = {}
            for concurrency, key in [(1, "quick"), (128, "full")]:
                passes = {
                    label: [
                        t
                        for t in report["trials"]
                        if t["concurrency"] == concurrency and t["mode"] == actual
                    ]
                    for label, actual in [("original", "original"), ("candidate", mode)]
                }
                comparison[f"c{concurrency}"] = {
                    "tokens_s": {
                        label: statistics.median(
                            t["prompt_tokens_per_second"] for t in ts
                        )
                        for label, ts in passes.items()
                    },
                    "latency_p50": {
                        label: statistics.median(
                            t["latency"]["p50_seconds"] for t in ts
                        )
                        for label, ts in passes.items()
                    },
                    "latency_p95": {
                        label: statistics.median(
                            t["latency"]["p95_seconds"] for t in ts
                        )
                        for label, ts in passes.items()
                    },
                    "paired_throughput": paired_ratios(
                        [t["prompt_tokens_per_second"] for t in passes["original"]],
                        [t["prompt_tokens_per_second"] for t in passes["candidate"]],
                    ),
                    "paired_scores": paired_score_summary(
                        values_by_workload[concurrency]["original"],
                        values_by_workload[concurrency][mode],
                    ),
                    "ranking": ranking_comparison(
                        workloads[key],
                        values_by_workload[concurrency]["original"],
                        values_by_workload[concurrency][mode],
                    ),
                }
            comparisons[mode] = comparison
        report.update(
            status="complete",
            comparisons=comparisons,
            selected=choose_placement(comparisons),
        )
        completed = True
    finally:
        # Any partial failure restores original placement, including partial mutations.
        mode = report["selected"] if completed else "original"
        masks = {}
        failures = []
        for label, process, mask in zip(
            ["api", "engine"], [api, worker], configurations[mode], strict=True
        ):
            try:
                masks[label] = process.apply(mask)
            except Exception as error:
                failures.append(error)
        if failures:
            # Do not let a dead/recycled API prevent restoration of the live engine.
            for process in [api, worker]:
                try:
                    process.apply(None)
                except Exception as error:
                    failures.append(error)
            report["placement_errors"] = [str(error) for error in failures]
            write(out / "summary.json", report)
            raise ExceptionGroup("failed to retain CPU placement", failures)
        report["retained_affinity"] = masks
        write(out / "summary.json", report)
    print("placement_complete", report["selected"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--node", required=True, type=int)
    args = parser.parse_args()
    out = ROOT / "results/b200_inference_benchmark" / args.name
    if Path(args.name).name != args.name or args.node < 0 or out.exists():
        raise ValueError("invalid run name or NUMA node")
    try:
        asyncio.run(measure(args.name, args.node))
    except BaseException as error:
        write(
            ROOT / "results/b200_inference_benchmark" / args.name / "failure.json",
            {"error": f"{type(error).__name__}: {error}"},
        )
        raise


if __name__ == "__main__":
    main()
