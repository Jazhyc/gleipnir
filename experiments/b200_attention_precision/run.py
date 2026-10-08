"""Run frozen BF16/FP8 projection-only screens and both descriptive ID passes."""

from __future__ import annotations

import argparse
import asyncio
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

from experiments.b200_attention_precision.worker import validate_native
from experiments.b200_inference_benchmark.run import DATA, ROOT, sha, write
from experiments.b200_long_context.run import gpu, rpc
from experiments.b200_monitor_score.run import SERVING, trial
from experiments.b200_optimized_id.analyze import compare
from experiments.b200_vllm031.id import bind_workload, require_stock
from experiments.b200_vllm031.run import archive_audits, may_benchmark
from experiments.b200_vllm031.runtime import candidate_environment
from gleipnir._compat import canonical_source_reference
from gleipnir.serving.benchmark import (
    measurement_summary,
    paired_score_summary,
    ranking_comparison,
)

EXPERIMENT = Path(__file__).parent
SOURCE_PATHS = [
    "src/gleipnir/serving/vllm/attention_precision.py",
    "experiments/b200_attention_gdn_serving/mixed_worker.py",
    "experiments/b200_attention_gdn_serving/attention_fp4_worker.py",
    *[str(p.relative_to(ROOT)) for p in EXPERIMENT.glob("*.py")],
]


def make_command(parent: list[str], precision: str, native: str) -> list[str]:
    """Change only full-attention projection scope and source-bound wrappers."""
    if precision not in {"bf16", "fp8"}:
        raise ValueError("unknown projection precision")
    command = parent.copy()
    command[0] = sys.executable
    command[command.index("-m") + 1] = "experiments.b200_attention_precision.server"
    command[command.index("--worker-cls") + 1] = (
        "experiments.b200_attention_precision.worker.AttentionPrecisionWorker"
    )
    command[command.index("--quantization") + 1] = "gleipnir_frost_attention_precision"
    index = command.index("--additional-config") + 1
    additional = json.loads(command[index])
    condition = additional["serving_condition"]
    if (
        condition["gdn_projection_precision"] != "fp4"
        or condition["attention_precision"] != "mxfp8"
    ):
        raise ValueError("non-target precision changed")
    condition["attention_projection_precision"] = precision
    condition["attention_precision_validation"] = native
    sources = {canonical_source_reference(p) for p in additional["gleipnir_frost_fp4"]}
    sources.update(SOURCE_PATHS)
    additional["gleipnir_frost_fp4"] = {p: sha(ROOT / p) for p in sorted(sources)}
    command[index] = json.dumps(additional, sort_keys=True)
    return command


async def canary(settings: dict, out: Path) -> None:
    rows = json.loads((DATA / "canary.json").read_text())
    cached = json.loads((ROOT / settings["cached_canary"]).read_text())
    master = json.loads((ROOT / settings["master_canary"]).read_text())
    if (
        master["master_sha256"] != settings["master_sha256"]
        or [r["prompt_sha256"] for r in rows] != master["prompt_sha256"]
    ):
        raise ValueError("master canary prompt/adapter identity changed")
    values, _ = await trial(rows, 4, settings)
    scores = np.array([r["score"] for r in values])
    references = {
        "accepted_024": np.array(cached["served"]["adapter"]),
        "master_bf16": np.array(master["adapter"]),
        "stock_031": np.array(
            [
                r["score"]
                for r in json.loads(
                    (
                        ROOT / settings["candidate"] / "canary_predictions.json"
                    ).read_text()
                )
            ]
        ),
    }
    drift = {
        k: {
            "mae": float(np.abs(scores - v).mean()),
            "correlation": float(np.corrcoef(scores, v)[0, 1]),
        }
        for k, v in references.items()
    }
    effect = float(np.abs(scores - np.array(master["base"])).max())
    accepted = drift["accepted_024"]
    receipt = {
        "passed": accepted["mae"] <= 0.005 and accepted["correlation"] >= 0.995,
        "finite": bool(np.isfinite(scores).all())
        and all(np.isfinite(r["margin"]) for r in values),
        "mean_absolute_difference": accepted["mae"],
        "correlation": accepted["correlation"],
        "adapter_effect": effect,
        "diagnostic_continuation_authorized": True,
        "references": drift,
        "strict_master_passed": drift["master_bf16"]["mae"] <= 0.02
        and drift["master_bf16"]["correlation"] >= 0.99,
    }
    write(out / "canary.json", receipt)
    write(out / "canary_predictions.json", values)
    if not may_benchmark(receipt, True):
        raise ValueError("precision candidate finite/adapter-effect guard failed")
    print("precision_canary", out.name, receipt, flush=True)


async def condition(
    name: str,
    precision: str,
    native_path: str,
    settings: dict,
    stock: dict,
    id_workload: list[dict],
    bf16_reference: list[dict],
    optimized_reference: list[dict],
    stock_id: list[dict],
) -> None:
    snapshot = gpu()
    if (
        snapshot["apps"]
        or settings["gpu_uuid"] not in snapshot["gpu"]
        or (SERVING / "server.json").exists()
    ):
        raise ValueError("precision run requires verified idle B200")
    out = ROOT / "results/b200_attention_precision" / name / precision
    out.mkdir(parents=True, exist_ok=False)
    command = make_command(stock["command"], precision, native_path)
    env = candidate_environment(ROOT)
    env["VLLM_GDN_DECODE_KERNEL"] = "cuda"
    env["GLEIPNIR_FLASHINFER_GDN_CP"] = "auto"
    env["GLEIPNIR_GIGATOKEN_RECEIPT"] = str(out / "frontend.json")
    parent = json.loads(
        (ROOT / settings["candidate"] / "parent_server.json").read_text()
    )
    env["GLEIPNIR_FROST_WRAPPER_VALIDATION"] = parent["host_wrapper"]["validation"]
    from vllm.v1.core.sched import scheduler

    scheduler_binding = {
        "upstream_sha256": sha(Path(inspect.getfile(scheduler))),
        "integration_sha256": sha(ROOT / "experiments/b200_vllm031/scheduler.py"),
    }
    if scheduler_binding != stock["scheduler_binding"]:
        raise ValueError("stock scheduler binding changed")
    env["GLEIPNIR_VLLM031_SCHEDULER_BINDING"] = json.dumps(scheduler_binding)
    log = (
        ROOT / "logs/runpod/b200_attention_precision" / f"{name}_{precision}_server.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    for path in SOURCE_PATHS:
        target = out / "executed_sources" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / path).read_bytes())
    write(out / "settings.json", settings)
    write(out / "initial_gpu.json", snapshot)
    report = {
        "status": "starting",
        "precision": precision,
        "promoted": False,
        "trials": [],
        "id_passes": [],
        "command": command,
        "source_hashes": {p: sha(ROOT / p) for p in SOURCE_PATHS},
        "native_sha256": sha(ROOT / native_path),
        "config_sha256": sha(EXPERIMENT / "config.json"),
        "runtime": stock["runtime"],
        "finite_canary_diagnostic": True,
    }
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
        "runtime_migration": stock["runtime"],
        "log": str(log),
        "frontend": {"receipt_path": str(out / "frontend.json")},
    }
    report["pid"] = process.pid
    write(out / "summary.json", report)
    write(out / "server.json", active)
    write(SERVING / "server.json", active)
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{settings['port']}", trust_env=False, timeout=30
        ) as client:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"precision server exited: {process.returncode}")
                try:
                    if (await client.get("/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.perf_counter() - started > 1200:
                    raise TimeoutError("precision startup exceeded twenty minutes")
                await asyncio.sleep(2)
        report.update(
            status="development", startup_seconds=time.perf_counter() - started
        )
        active["status"] = "ready"
        write(out / "summary.json", report)
        write(out / "server.json", active)
        write(SERVING / "server.json", active)
        print("precision_ready", precision, report["startup_seconds"], flush=True)
        if (await rpc("frost_wrapper_state"))["mode"] != "direct":
            raise ValueError("direct FROST host binding changed")
        await canary(settings, out)
        archive_audits(out)
        scope = json.loads((out / "native_attention_projections.json").read_text())
        if (
            not scope["passed"]
            or scope["worker_pid"]
            != json.loads((out / "loaded_precision.json").read_text())["worker_pid"]
        ):
            raise ValueError("actual projection execution scope is incomplete")
        for concurrency, key, repeats in ((1, "quick", 3), (128, "full", 6)):
            workload = json.loads((DATA / f"{key}.json").read_text())
            warm, _ = await trial(workload, concurrency, settings)
            write(out / f"c{concurrency}_warmup.json", warm)
            observations = []
            for repeat in range(repeats):
                values, seconds = await trial(workload, concurrency, settings)
                observations.append(values)
                write(out / f"c{concurrency}_repeat{repeat}.json", values)
                report["trials"].append(
                    {
                        "concurrency": concurrency,
                        "repeat": repeat,
                        **measurement_summary(values, seconds),
                    }
                )
                write(out / "summary.json", report)
                print(
                    "precision_dev_pass",
                    precision,
                    concurrency,
                    repeat,
                    seconds,
                    flush=True,
                )
            archived = [
                json.loads(
                    (
                        ROOT / settings["candidate"] / f"c{concurrency}_repeat{i}.json"
                    ).read_text()
                )
                for i in range(repeats)
            ]
            write(
                out / f"c{concurrency}_comparison.json",
                {
                    "scores": paired_score_summary(archived, observations),
                    "ranking": ranking_comparison(workload, archived, observations),
                },
            )
        archive_audits(out)
        report["status"] = "id"
        write(out / "summary.json", report)
        observed = []
        started = time.perf_counter()
        for offset in range(0, len(id_workload), 128):
            values, seconds = await trial(
                id_workload[offset : offset + 128], 128, settings
            )
            observed.extend(values)
            write(out / f"id/batch{offset // 128:03d}.json", values)
            print(
                "precision_id_progress",
                precision,
                len(observed),
                len(id_workload),
                flush=True,
            )
        seconds = time.perf_counter() - started
        write(out / "id_predictions.json", observed)
        report["id_passes"].append(
            {
                "repeat": 0,
                "first_id_shape_use_inclusive": True,
                **measurement_summary(observed, seconds),
            }
        )
        result = compare(id_workload, bf16_reference, [observed])
        result["vs_optimized_024"] = compare(
            id_workload, optimized_reference, [observed]
        )
        result["vs_fp4_031"] = compare(id_workload, stock_id, [observed])
        result["analysis_source_sha256"] = sha(Path(inspect.getfile(compare)))
        (out / "analysis_source.py").write_bytes(
            Path(inspect.getfile(compare)).read_bytes()
        )
        write(out / "id_comparison.json", result)
        archive_audits(out)
        report["status"] = "complete"
        write(out / "summary.json", report)
        print(
            "precision_complete",
            precision,
            seconds,
            result["vs_fp4_031"]["metric_deltas"],
            flush=True,
        )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write(out / "summary.json", report)
        raise
    finally:
        if process.poll() is None:
            if (SERVING / "loaded_precision.json").exists():
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "experiments.b200_vllm031.stop",
                        "--archive-name",
                        f"{name}_{precision}",
                    ],
                    cwd=ROOT,
                    env=env,
                    check=True,
                )
                retired = (
                    ROOT / "results/b200_vllm031" / f"{name}_{precision}_retired.json"
                )
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
                    raise ValueError(
                        "failed precision startup process identity changed"
                    )
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=45)
                (SERVING / "server.json").unlink()
        elif not gpu()["apps"] and (SERVING / "server.json").exists():
            if json.loads((SERVING / "server.json").read_text())["pid"] == process.pid:
                (SERVING / "server.json").unlink()
        write(out / "closure.json", gpu())


async def run(name: str, native_path: str) -> None:
    if not name or Path(name).name != name:
        raise ValueError("campaign name must be a stem")
    settings = json.loads((EXPERIMENT / "config.json").read_text())
    for path, expected in settings["files_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"frozen experiment artifact changed: {path}")
    native = json.loads((ROOT / native_path).read_text())
    for precision in ("bf16", "fp8"):
        validate_native(native, precision)
    stock = json.loads((ROOT / settings["candidate"] / "summary.json").read_text())
    require_stock(stock)
    if {n: importlib.metadata.version(n) for n in stock["runtime"]} != stock["runtime"]:
        raise ValueError("stock runtime changed")
    control = ROOT / settings["optimized_control"]
    workload = json.loads((control / "workload.json").read_text())
    reference = json.loads((control / "reference.json").read_text())
    optimized = bind_workload(
        workload, reference, json.loads((control / "repeat0.json").read_text())
    )
    fp4 = bind_workload(
        workload,
        reference,
        json.loads((ROOT / settings["stock_id"] / "repeat0.json").read_text()),
    )
    merge = json.loads((control / "merged_artifact.json").read_text())
    model = Path(settings["merged_model"])
    if (
        json.loads((model / "merge_manifest.json").read_text()) != merge
        or merge["adapter_sha256"] != settings["serving_adapter_sha256"]
    ):
        raise ValueError("merged trained adapter identity changed")
    for filename, expected in merge["files_sha256"].items():
        if sha(model / filename) != expected:
            raise ValueError(f"merged model source changed: {filename}")
    for field in ("master", "serving_adapter"):
        if sha(ROOT / settings[field]) != settings[field + "_sha256"]:
            raise ValueError(f"trained adapter source changed: {field}")
    out = ROOT / "results/b200_attention_precision" / name
    out.mkdir(parents=True, exist_ok=False)
    write(out / "settings.json", settings)
    write(out / "native.json", native)
    write(out / "id_reference.json", reference)
    write(out / "id_workload.json", workload)
    report = {
        "status": "running",
        "conditions": ["bf16", "fp8"],
        "completed": [],
        "promoted": False,
    }
    write(out / "campaign.json", report)
    try:
        for precision in ("bf16", "fp8"):
            await condition(
                name,
                precision,
                native_path,
                settings,
                stock,
                workload,
                reference,
                optimized,
                fp4,
            )
            report["completed"].append(precision)
            write(out / "campaign.json", report)
        report["status"] = "complete"
        write(out / "campaign.json", report)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write(out / "campaign.json", report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--native-receipt", required=True)
    args = parser.parse_args()
    asyncio.run(run(args.name, args.native_receipt))


if __name__ == "__main__":
    main()
