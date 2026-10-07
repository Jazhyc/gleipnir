"""Retry the full sweep after the recorded exact-cap pooling scheduling stall."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import inspect
import json
import subprocess
import time
from pathlib import Path

import httpx

from experiments.b200_inference_benchmark.run import ROOT, sha, write
from experiments.b200_inference_benchmark.run import environment as base_environment
from experiments.b200_long_context.envelope import validate
from experiments.b200_long_context.run import (
    EXPERIMENT,
    SERVING,
    canary,
    gpu,
    measure_lengths,
    rpc,
)
from gleipnir.serving.score_runtime import resume_score_environment


async def run(name: str, failed_name: str) -> None:
    old = ROOT / "results/b200_long_context" / failed_name
    failure = json.loads((old / "summary.json").read_text())
    if failure["status"] != "failed" or not (old / "boundary_stall.json").exists():
        raise ValueError("recorded exact-cap stall is required for this retry")
    if (SERVING / "server.json").exists() or gpu()["apps"]:
        raise ValueError("pooling-boundary retry requires an idle GPU")
    out = ROOT / "results/b200_long_context" / name
    out.mkdir(parents=True, exist_ok=False)
    settings = json.loads((old / "settings.json").read_text())
    rows = json.loads((old / "workload.json").read_text())
    binding = json.loads((old / "workload_binding.json").read_text())
    if sha(old / "workload.json") != binding["sha256"]:
        raise ValueError("retry workload changed")
    for row in rows:
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
            raise ValueError("retry prompt hash changed")
    for filename in (
        "settings.json",
        "workload.json",
        "workload_binding.json",
        "native.json",
        "quality.json",
    ):
        (out / filename).write_bytes((old / filename).read_bytes())
    validate(json.loads((out / "native.json").read_text()), ROOT)
    for p in EXPERIMENT.iterdir():
        if p.suffix in {".py", ".json", ".md"}:
            destination = out / "executed_sources" / p.name
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(p.read_bytes())
    parent = json.loads((old / "server.json").read_text())
    write(out / "parent_server.json", parent)
    report = {
        "status": "starting",
        "measurements": [],
        "promoted": False,
        "failure_parent": failed_name,
        "unchanged_native_quality_reuse": {
            str((old / f).relative_to(ROOT)): sha(old / f)
            for f in ("native.json", "quality.json")
        },
    }
    write(out / "summary.json", report)
    from vllm.v1.core.sched import scheduler

    cpu_binding = {
        "scheduler_sha256": sha(Path(inspect.getfile(scheduler))),
        "integration_sha256": sha(EXPERIMENT / "scheduler.py"),
        "runner": "pooling",
        "generated_token_reservation": 0,
        "queue_policy": "unchanged_fcfs",
    }
    write(out / "pooling_boundary.json", cpu_binding)
    base = base_environment()
    base["PYTHONPATH"] = f"/tmp/gleipnir-serving-source-bootstrap:{base['PYTHONPATH']}"
    env = resume_score_environment(ROOT, parent, base)
    env["GLEIPNIR_LONG_CONTEXT_VALIDATION"] = str(
        (out / "native.json").relative_to(ROOT)
    )
    env["GLEIPNIR_POOLING_BOUNDARY_CONFIG"] = json.dumps(cpu_binding, sort_keys=True)
    env["GLEIPNIR_GIGATOKEN_RECEIPT"] = str(out / "frontend.json")
    command = parent["command"] + [
        "--scheduler-cls",
        "experiments.b200_long_context.scheduler.PoolingContextScheduler",
    ]
    process = None
    try:
        started = time.perf_counter()
        log = ROOT / "logs/runpod/b200_attention_gdn_serving/server.log"
        with log.open("x") as handle:
            process = subprocess.Popen(
                command,
                env=env,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        server = copy.deepcopy(parent)
        server.update(
            pid=process.pid,
            command=command,
            status="starting",
            parent_pid=parent["pid"],
            started_at_unix=time.time(),
            long_context_validation=str((out / "native.json").relative_to(ROOT)),
            pooling_boundary=cpu_binding,
        )
        server["frontend"]["receipt_path"] = str(out / "frontend.json")
        write(SERVING / "server.json", server)
        write(out / "server.json", server)
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8010", trust_env=False, timeout=30
        ) as client:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"boundary-corrected server exited: {process.returncode}"
                    )
                try:
                    if (await client.get("/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.perf_counter() - started > settings["startup_timeout_seconds"]:
                    raise TimeoutError("boundary-corrected startup timed out")
                await asyncio.sleep(2)
        from gleipnir.serving_cache_mirror import persist_compiler_mirror

        server.update(status="ready", ready_at_unix=time.time())
        server["compiler_cache_persistence"] = persist_compiler_mirror(
            server.get("compiler_mirror")
        )
        write(SERVING / "server.json", server)
        write(out / "server.json", server)
        report["startup_seconds"] = time.perf_counter() - started
        print(
            "boundary_corrected_server_ready",
            process.pid,
            report["startup_seconds"],
            flush=True,
        )
        if (await rpc("frost_wrapper_state"))["mode"] != "direct":
            raise ValueError("direct FROST binding changed")
        await canary(settings, out, "extended")
        report["status"] = "running_extended"
        await measure_lengths(rows, settings, out, report, "extended")
        for filename in (
            "monitor_score.json",
            "loaded_precision.json",
            "compile_identity.json",
            "native_attention.json",
            "native_preparation.json",
            "native_gemm_tuning.json",
            "native_attention_projections.json",
            "native_swiglu_output.json",
        ):
            write(out / filename, json.loads((SERVING / filename).read_text()))
        write(
            out / "closure.json", {**gpu(), "memory": await rpc("long_context_memory")}
        )
        report["status"] = "complete"
        write(out / "summary.json", report)
        print("long_context_retry_complete", flush=True)
    except BaseException as error:
        report.update(
            status="failed", error=f"{type(error).__name__}: {error}", gpu=gpu()
        )
        write(out / "summary.json", report)
        if process is not None and process.poll() is None:
            subprocess.run(
                [
                    command[0],
                    "-m",
                    "experiments.b200_attention_gdn_serving.stop_server",
                    "--archive-name",
                    name + "_failed",
                    "--reason",
                    report["error"],
                ],
                env=env,
                cwd=ROOT,
                check=True,
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--failed-name", required=True)
    args = parser.parse_args()
    if any(
        x in {"", ".", ".."} or Path(x).name != x for x in (args.name, args.failed_name)
    ):
        raise ValueError("run names must be stems")
    asyncio.run(run(args.name, args.failed_name))


if __name__ == "__main__":
    main()
