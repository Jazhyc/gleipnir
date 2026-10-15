"""Torch-profile the adopted last-query default at C1 lengths and a c128 batch."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import subprocess
import time
from pathlib import Path

import httpx

from experiments.b200_context_million.environment import snapshot
from experiments.b200_context_million.run import (
    canary,
    phase_command,
    retire,
    server_environment,
    wait_ready,
)
from experiments.b200_inference_benchmark.run import DATA, ROOT, sha, write
from experiments.b200_kernel_profile.analyze import summarize
from experiments.b200_last_layer_trim.run import LAST
from experiments.b200_long_context.run import gpu, rpc
from experiments.b200_monitor_score.run import SERVING, trial
from experiments.b200_vllm031.source_bindings import restore_diagnostic_sources
from gleipnir.serving.reference import selected_serving_default

EXPERIMENT = Path(__file__).parent


async def profiled(
    rows: list[dict], concurrency: int, settings: dict, out: Path
) -> dict:
    """One excluded warmup, then exactly one traced pass."""
    profiles = SERVING / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    await trial(rows, concurrency, settings)
    existing = {p.name for p in profiles.glob("*.gz")}
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{settings['port']}", trust_env=False, timeout=600
    ) as client:
        (await client.post("/start_profile")).raise_for_status()
        try:
            values, seconds = await trial(rows, concurrency, settings)
        finally:
            (await client.post("/stop_profile")).raise_for_status()
    for _ in range(600):
        new = [p for p in profiles.glob("*.gz") if p.name not in existing]
        if new:
            break
        time.sleep(1)
    if len(new) != 1:
        raise ValueError(f"unexpected profile exports: {len(new)}")
    time.sleep(5)  # let the exporter finish writing the compressed trace
    out.mkdir(parents=True, exist_ok=False)
    trace = out / new[0].name
    trace.write_bytes(new[0].read_bytes())
    new[0].unlink()
    write(out / "predictions.json", values)
    report = {
        **summarize(trace),
        "instrumented_seconds": seconds,
        "requests": len(values),
        "input_tokens": sum(r["prompt_tokens"] for r in values),
        "concurrency": concurrency,
    }
    write(out / "summary.json", report)
    return report


async def run(name: str, phases: list[str] | None = None) -> None:
    settings = json.loads((EXPERIMENT / "config.json").read_text())
    out = ROOT / "results/b200_kernel_profile" / name
    out.mkdir(parents=True, exist_ok=False)
    write(out / "settings.json", settings)
    for p in EXPERIMENT.iterdir():
        if p.suffix in {".py", ".json", ".md"}:
            destination = out / "executed_sources" / p.name
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(p.read_bytes())
    write(out / "environment_start.json", snapshot(ROOT, raw_directory=out / "env"))
    if gpu()["apps"] or (SERVING / "server.json").exists():
        raise ValueError("profiling requires an idle GPU and no server")
    selection, command = selected_serving_default(ROOT, last_query=False)
    from vllm.v1.core.sched import scheduler

    binding = {
        "upstream_sha256": sha(Path(inspect.getfile(scheduler))),
        "integration_sha256": sha(ROOT / "experiments/b200_vllm031/scheduler.py"),
    }
    if binding != selection["scheduler_binding"]:
        raise ValueError("accepted pooling scheduler source changed")
    # Archived diagnostic scripts must match their evaluated bytes before launch.
    archive = ROOT / "results/b200_vllm031/pre_migration_sources.tar.gz"
    additional = json.loads(command[command.index("--additional-config") + 1])
    write(
        out / "source_restoration.json",
        {
            "archive_sha256": sha(archive),
            "restored": restore_diagnostic_sources(
                ROOT, additional["serving_condition"], archive
            ),
        },
    )
    workload = {
        r["prompt_tokens"]: r
        for r in json.loads((ROOT / settings["workload"]).read_text())
    }
    native = ROOT / settings["long_context_validation"]
    report = {"status": "running", "cases": {}}
    for phase in settings["phases"]:
        if phases and phase["name"] not in phases:
            continue
        phase_out = out / phase["name"]
        phase_out.mkdir()
        server = phase_command(command, phase, settings["chunk_tokens"], LAST)
        env = server_environment(selection, phase_out, binding, native)
        env["GLEIPNIR_LAST_QUERY_VALIDATION"] = settings["last_query_validation"]
        write(phase_out / "command.json", server)
        log = ROOT / "logs/runpod/b200_kernel_profile" / f"{name}_{phase['name']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as handle:
            process = subprocess.Popen(
                server,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        active = {
            "pid": process.pid,
            "command": server,
            "status": "starting",
            "runtime_migration": selection["runtime"],
            "log": str(log),
        }
        write(SERVING / "server.json", active)
        try:
            await wait_ready(process, settings["startup_timeout_seconds"])
            await canary(selection, settings, phase_out, phase["canary_gate"])
            for length in phase["lengths"]:
                case = f"{phase['name']}_c1_{length}"
                report["cases"][case] = await profiled(
                    [workload[length]], 1, settings, phase_out / f"c1_{length}"
                )
                write(out / "summary.json", report)
                print("profiled", case, flush=True)
            if phase["batched"]:
                full = json.loads((DATA / "full.json").read_text())
                case = f"{phase['name']}_c128_full320"
                report["cases"][case] = await profiled(
                    full, 128, settings, phase_out / "c128_full320"
                )
                write(out / "summary.json", report)
                print("profiled", case, flush=True)
            write(phase_out / "last_query_audit.json", await rpc("last_query_audit"))
        finally:
            if process.poll() is None:
                retire(f"{name}_{phase['name']}", env)
            elif (SERVING / "server.json").exists():
                (SERVING / "server.json").unlink()
    report["status"] = "complete"
    write(out / "summary.json", report)
    print("kernel_profile_complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--phase", action="append", choices=["native", "yarn"])
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("run name must be a stem")
    asyncio.run(run(args.name, args.phase))


if __name__ == "__main__":
    main()
