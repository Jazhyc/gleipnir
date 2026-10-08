"""Evaluate the frozen full-trained adapter on ID with the migrated stock recipe."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from experiments.b200_inference_benchmark.run import DATA, ROOT, sha, write
from experiments.b200_long_context.run import gpu, rpc
from experiments.b200_monitor_score.run import SERVING, trial
from experiments.b200_optimized_id.analyze import compare
from experiments.b200_vllm031.run import archive_audits, may_benchmark
from experiments.b200_vllm031.runtime import candidate_environment
from gleipnir.serving.benchmark import measurement_summary

EXPERIMENT = Path(__file__).parent


def require_stock(recipe: dict) -> None:
    """Reject optional ablations when reproducing the frozen stock condition."""
    expected = {
        "status": "complete",
        "scheduler_mode": "default",
        "gdn_decode_kernel": "cuda",
        "gdn_cp": "auto",
        "enforce_eager": False,
        "max_num_active_seqs": None,
    }
    if any(recipe.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen stock migration recipe changed")


def bind_workload(
    workload: list[dict], reference: list[dict], optimized: list[dict]
) -> list[dict]:
    """Check ordered labels, prompt content, usage and both score references."""
    if not (len(workload) == len(reference) == len(optimized) == 3012):
        raise ValueError("frozen ID population changed")
    if len({r["id"] for r in workload}) != len(workload):
        raise ValueError("duplicate ID identity")
    paired = []
    for row, old, quant in zip(workload, reference, optimized, strict=True):
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
            raise ValueError("frozen rendered prompt content changed")
        for key in ("id", "prompt_sha256", "prompt_tokens"):
            if row[key] != old[key] or row[key] != quant[key]:
                raise ValueError(f"ID prediction binding changed: {key}")
        if row["label"] != old["label"] or row["dataset"] != old["dataset"]:
            raise ValueError("ID label/source binding changed")
        if not 0 < row["prompt_tokens"] < 32768:
            raise ValueError("ID prompt outside frozen context envelope")
        paired.append(old | quant)
    return paired


async def check_canary(settings: dict, out: Path) -> None:
    rows = json.loads((DATA / "canary.json").read_text())
    candidate = ROOT / settings["candidate"]
    previous = json.loads((candidate / "canary_predictions.json").read_text())
    cached = json.loads((ROOT / settings["cached_canary"]).read_text())
    master = json.loads((ROOT / settings["master_canary"]).read_text())
    if (
        master["master_sha256"] != settings["master_sha256"]
        or [r["prompt_sha256"] for r in rows] != master["prompt_sha256"]
    ):
        raise ValueError("master canary source binding changed")
    observed, _ = await trial(rows, 4, settings)
    if [(r["id"], r["prompt_sha256"]) for r in observed] != [
        (r["id"], r["prompt_sha256"]) for r in previous
    ]:
        raise ValueError("migration canary identities changed")
    scores = np.array([r["score"] for r in observed])

    def drift(target):
        return {
            "mae": float(np.abs(scores - target).mean()),
            "correlation": float(np.corrcoef(scores, target)[0, 1]),
        }

    accepted = drift(np.array(cached["served"]["adapter"]))
    reproduced = drift(np.array([r["score"] for r in previous]))
    effect = float(np.abs(scores - np.array(master["base"])).max())
    mean_limit = settings["canary_mean_error_limit"]
    corr_floor = settings["canary_correlation_floor"]
    receipt = {
        "passed": accepted["mae"] <= mean_limit
        and accepted["correlation"] >= corr_floor,
        "finite": bool(np.isfinite(scores).all())
        and all(np.isfinite(r["margin"]) for r in observed),
        "mean_absolute_difference": accepted["mae"],
        "correlation": accepted["correlation"],
        "adapter_effect": effect,
        "cached_recipe": accepted,
        "master": drift(np.array(master["adapter"])),
        "migration_reproduction": reproduced,
        "migration_reproduction_passed": reproduced["mae"] <= mean_limit
        and reproduced["correlation"] >= corr_floor,
        "diagnostic_continuation_authorized": settings["finite_canary_diagnostic"],
    }
    write(out / "canary.json", receipt)
    write(out / "canary_predictions.json", observed)
    if (
        not may_benchmark(receipt, settings["finite_canary_diagnostic"])
        or not receipt["migration_reproduction_passed"]
    ):
        raise ValueError("finite/reproduction canary failed")
    print("id_canary", receipt, flush=True)


async def run(name: str) -> None:
    """Run one grouped diagnostic pass and retire only its owned server."""
    if Path(name).name != name or not name:
        raise ValueError("run name must be a stem")
    settings = json.loads((EXPERIMENT / "id_config.json").read_text())
    if settings["repeats"] != 1 or settings["promote"]:
        raise ValueError("ID run must remain one-pass and unpromoted")
    for filename, expected in settings["files_sha256"].items():
        if sha(ROOT / filename) != expected:
            raise ValueError(f"frozen artifact changed: {filename}")
    parent = ROOT / settings["candidate"]
    frozen = json.loads((parent / "summary.json").read_text())
    require_stock(frozen)
    runtime = {n: importlib.metadata.version(n) for n in frozen["runtime"]}
    if runtime != frozen["runtime"]:
        raise ValueError("migration runtime changed")
    for filename, expected in frozen["source_hashes"].items():
        # The diagnostic launcher later gained active-admission trials. Its
        # shared guards are bound to current bytes in id_config; server code
        # must still match the original stock run exactly.
        if filename == "experiments/b200_vllm031/run.py":
            continue
        if sha(ROOT / filename) != expected:
            raise ValueError(f"migration runtime source changed: {filename}")
    control = ROOT / settings["optimized_control"]
    workload = json.loads((control / "workload.json").read_text())
    reference = json.loads((control / "reference.json").read_text())
    optimized = bind_workload(
        workload, reference, json.loads((control / "repeat0.json").read_text())
    )
    merge = json.loads((control / "merged_artifact.json").read_text())
    model = Path(settings["merged_model"])
    if (
        json.loads((model / "merge_manifest.json").read_text()) != merge
        or merge["adapter_sha256"] != settings["serving_adapter_sha256"]
        or sha(ROOT / settings["master"]) != settings["master_sha256"]
        or sha(ROOT / settings["serving_adapter"]) != settings["serving_adapter_sha256"]
    ):
        raise ValueError("trained adapter or merged source identity changed")
    for filename, expected in merge["files_sha256"].items():
        if sha(model / filename) != expected:
            raise ValueError(f"merged checkpoint changed: {filename}")
    snapshot = gpu()
    if (
        snapshot["apps"]
        or settings["gpu_uuid"] not in snapshot["gpu"]
        or (SERVING / "server.json").exists()
    ):
        raise ValueError("ID evaluation requires the verified idle B200")
    out = ROOT / "results/b200_vllm031" / name
    out.mkdir(parents=True, exist_ok=False)
    for filename, value in {
        "settings.json": settings,
        "initial_gpu.json": snapshot,
        "workload.json": workload,
        "reference.json": reference,
        "optimized_reference.json": optimized,
        "merged_artifact.json": merge,
    }.items():
        write(out / filename, value)
    (out / "executed_id.py").write_bytes(Path(__file__).read_bytes())
    command = frozen["command"].copy()
    command[0] = sys.executable
    if command[command.index("--model") + 1] != str(model):
        raise ValueError("stock recipe checkpoint changed")
    env = candidate_environment(ROOT)
    env["VLLM_GDN_DECODE_KERNEL"] = frozen["gdn_decode_kernel"]
    env["GLEIPNIR_FLASHINFER_GDN_CP"] = frozen["gdn_cp"]
    env["GLEIPNIR_GIGATOKEN_RECEIPT"] = str(out / "frontend.json")
    original = json.loads((parent / "parent_server.json").read_text())
    env["GLEIPNIR_FROST_WRAPPER_VALIDATION"] = original["host_wrapper"]["validation"]
    from vllm.v1.core.sched import scheduler

    binding = {
        "upstream_sha256": sha(Path(inspect.getfile(scheduler))),
        "integration_sha256": sha(EXPERIMENT / "scheduler.py"),
    }
    if binding != frozen["scheduler_binding"]:
        raise ValueError("stock scheduler sources changed")
    env["GLEIPNIR_VLLM031_SCHEDULER_BINDING"] = json.dumps(binding)
    log = ROOT / "logs/runpod/b200_vllm031" / f"{name}_server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("x") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    active = {
        "pid": process.pid,
        "command": command,
        "status": "starting",
        "runtime_migration": runtime,
        "log": str(log),
        "frontend": {"receipt_path": str(out / "frontend.json")},
    }
    write(SERVING / "server.json", active)
    write(out / "server.json", active)
    report = {
        "status": "starting",
        "passes": [],
        "promoted": False,
        "finite_canary_diagnostic": True,
        "runtime": runtime,
        "pid": process.pid,
        "command": command,
        "config_sha256": sha(EXPERIMENT / "id_config.json"),
        "server_log": str(log.relative_to(ROOT)),
    }
    write(out / "summary.json", report)
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{settings['port']}",
            trust_env=False,
            timeout=30,
        ) as client:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"ID server exited: {process.returncode}")
                try:
                    if (await client.get("/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.perf_counter() - started > 1200:
                    raise TimeoutError("ID server startup exceeded twenty minutes")
                await asyncio.sleep(2)
        report["startup_seconds"] = time.perf_counter() - started
        active["status"] = "ready"
        write(SERVING / "server.json", active)
        write(out / "server.json", active)
        print("id_server_ready", report["startup_seconds"], flush=True)
        if (await rpc("frost_wrapper_state"))["mode"] != "direct":
            raise ValueError("direct FROST host binding changed")
        archive_audits(out)
        await check_canary(settings, out)
        warm, warm_seconds = await trial(
            json.loads((DATA / "quick.json").read_text()),
            settings["concurrency"],
            settings,
        )
        write(out / "training_warmup.json", {"seconds": warm_seconds, "values": warm})
        report["status"] = "running"
        write(out / "summary.json", report)
        values, batches = [], []
        started = time.perf_counter()
        for offset in range(0, len(workload), settings["batch_rows"]):
            batch = workload[offset : offset + settings["batch_rows"]]
            observed, seconds = await trial(batch, settings["concurrency"], settings)
            values.extend(observed)
            index = offset // settings["batch_rows"]
            write(out / f"repeat0/batch{index:03d}.json", observed)
            batches.append(
                {"batch": index, "rows": len(batch), "http_seconds": seconds}
            )
            print("id_progress", len(values), len(workload), flush=True)
        seconds = time.perf_counter() - started
        write(out / "repeat0.json", values)
        report["passes"].append(
            {
                "repeat": 0,
                "first_id_shape_use_inclusive": True,
                "batch_records": batches,
                **measurement_summary(values, seconds),
            }
        )
        report["status"] = "analyzing"
        write(out / "summary.json", report)
        result = compare(workload, reference, [values])
        result["vs_optimized_024"] = compare(workload, optimized, [values])
        result["input_artifact_sha256"] = {
            p.name: sha(p)
            for p in [
                out / "workload.json",
                out / "reference.json",
                out / "optimized_reference.json",
                out / "repeat0.json",
            ]
        }
        result["analysis_source_sha256"] = sha(Path(inspect.getfile(compare)))
        (out / "analysis_source.py").write_bytes(
            Path(inspect.getfile(compare)).read_bytes()
        )
        write(out / "comparison.json", result)
        archive_audits(out)
        report["status"] = "complete"
        write(out / "summary.json", report)
        print("id_complete", seconds, result["metric_deltas"], flush=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write(out / "summary.json", report)
        raise
    finally:
        if process.poll() is None:
            if (out / "loaded_precision.json").exists():
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "experiments.b200_vllm031.stop",
                        "--archive-name",
                        name,
                    ],
                    cwd=ROOT,
                    env=env,
                    check=True,
                )
                retired = ROOT / "results/b200_vllm031" / f"{name}_retired.json"
                write(out / "retirement.json", json.loads(retired.read_text()))
            else:
                actual = [
                    v.decode()
                    for v in Path(f"/proc/{process.pid}/cmdline")
                    .read_bytes()
                    .split(b"\0")
                    if v
                ]
                if actual != command or os.getpgid(process.pid) != process.pid:
                    raise ValueError("failed startup process identity changed")
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=45)
                (SERVING / "server.json").unlink()
        write(out / "closure.json", gpu())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    asyncio.run(run(parser.parse_args().name))


if __name__ == "__main__":
    main()
