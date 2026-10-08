"""Launch an isolated runtime and compare the frozen monitoring workload."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import inspect
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import yaml

from experiments.b200_inference_benchmark.run import DATA, ROOT, sha, write
from experiments.b200_monitor_score.run import trial
from experiments.b200_vllm031.runtime import candidate_environment
from experiments.b200_vllm031.source_bindings import restore_diagnostic_sources
from gleipnir._compat import canonical_source_reference
from gleipnir.serving.benchmark import (
    measurement_summary,
    paired_score_summary,
    ranking_comparison,
)
from gleipnir.serving.reference import selected_score_reference

EXPERIMENT = Path(__file__).parent


def may_benchmark(canary: dict, allow_diagnostic: bool) -> bool:
    """A diagnostic can retain failed agreement, never nonfinite/no-effect output."""
    return bool(
        canary["finite"]
        and canary["adapter_effect"] > 0
        and np.isfinite(canary["correlation"])
        and np.isfinite(canary["mean_absolute_difference"])
        and (canary["passed"] or allow_diagnostic)
    )


def archive_audits(out: Path) -> None:
    """Preserve loaded/native receipts on successful and failed score screens."""
    for filename in (
        "monitor_score.json",
        "loaded_precision.json",
        "compile_identity.json",
        "native_attention.json",
        "native_preparation.json",
        "native_gemm_tuning.json",
        "native_attention_projections.json",
        "native_swiglu_output.json",
        "migration_gdn_backend.json",
    ):
        source = ROOT / "results/b200_attention_gdn_serving" / filename
        if source.exists():
            write(out / filename, json.loads(source.read_text()))


def candidate_command(
    parent: list[str],
    native_receipt: str,
    vendor_receipt: str,
    mode: str,
    gdn_decode_kernel: str,
    gdn_cp: str,
    enforce_eager: bool,
    max_active: int | None,
) -> list[str]:
    """Change the runtime and cache adapter while preserving the scoring recipe."""
    command = parent.copy()
    command[0] = sys.executable
    command[command.index("-m") + 1] = "experiments.b200_vllm031.server"
    command[command.index("--worker-cls") + 1] = (
        "experiments.b200_vllm031.worker.MigrationWorker"
    )
    for flag in ("--scheduler-cls", "--long-prefill-token-threshold"):
        if flag in command:
            index = command.index(flag)
            del command[index : index + 2]
    for flag in ("--async-scheduling", "--no-async-scheduling"):
        if flag in command:
            command.remove(flag)
    index = command.index("--additional-config") + 1
    additional = json.loads(command[index])
    additional.pop("qk_mutation_analysis", None)
    additional["serving_condition"]["mxfp8_validation"] = native_receipt
    additional["serving_condition"]["fp4_prepare_validation"]["vendor"] = vendor_receipt
    additional["runtime_migration"] = {
        "vllm": "0.31.0",
        "gdn_decode_kernel": gdn_decode_kernel,
        "gdn_cp": gdn_cp,
        "backend_adapter_sha256": sha(EXPERIMENT / "backend_config.py"),
        "enforce_eager": enforce_eager,
    }
    sources = additional["gleipnir_frost_fp4"]
    additional["gleipnir_frost_fp4"] = {
        canonical_source_reference(p): sha(ROOT / canonical_source_reference(p))
        for p in sources
    }
    command[index] = json.dumps(additional, sort_keys=True)
    command += [
        "--no-async-scheduling",
        "--scheduler-cls",
        "experiments.b200_vllm031.scheduler.Vllm031PoolingScheduler",
    ]
    if mode == "adaptive":
        command += [
            "--long-prefill-token-threshold",
            "4096",
            "--long-prefill-token-threshold-adaptive",
        ]
    elif mode != "default":
        raise ValueError("unknown scheduler mode")
    if enforce_eager:
        command.append("--enforce-eager")
    if max_active is not None:
        if not 1 <= max_active <= int(command[command.index("--max-num-seqs") + 1]):
            raise ValueError("active admission must fit the unchanged runner capacity")
        command += ["--max-num-active-seqs", str(max_active)]
    return command


async def run(
    name: str,
    native_receipt: str,
    vendor_receipt: str,
    mode: str,
    gdn_decode_kernel: str,
    gdn_cp: str,
    gdn_receipt: str | None,
    enforce_eager: bool,
    allow_diagnostic: bool,
    max_active: int | None,
) -> None:
    if importlib.metadata.version("vllm") != "0.31.0":
        raise ValueError("run under the candidate vLLM 0.31 environment")
    # Exercise the reporting dependencies before allocating an expensive engine.
    import pandas as pd

    from gleipnir.evaluation.binary import metric_views

    metric_views(
        pd.DataFrame(
            {
                "dataset": ["preflight", "preflight"],
                "label": [0, 1],
                "score": [0.2, 0.8],
            }
        )
    )
    native = json.loads((ROOT / native_receipt).read_text())
    if not native.get("migration_passed"):
        raise ValueError("native runtime/cache/Qwen validation is incomplete")
    vendor = json.loads((ROOT / vendor_receipt).read_text())
    if not vendor.get("arithmetic_passed") or len(vendor["checks"]) != 12:
        raise ValueError("FlashInfer 0.7 CUDA packing validation is incomplete")
    if gdn_cp == "off":
        if gdn_receipt is None:
            raise ValueError("non-CP routing requires the paired native GDN receipt")
        gdn = json.loads((ROOT / gdn_receipt).read_text())
        if (
            len(gdn["checks"]) != 4
            or not gdn["qk"]["effective_weight_exact"]
            or any(
                not check[field]["finite"] or check[field]["relative_l2"] != 0
                for check in gdn["checks"]
                for field in ("no_cp_output", "no_cp_state")
            )
        ):
            raise ValueError("non-CP GDN does not reproduce the preserved kernel")
    selection = selected_score_reference(ROOT)
    parent = json.loads((ROOT / selection["server_metadata"]).read_text())
    gpu_apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if gpu_apps:
        raise ValueError("retire the identity-verified server before this launch")
    active_path = ROOT / "results/b200_attention_gdn_serving/server.json"
    if active_path.exists():
        raise ValueError("preserve or retire existing serving metadata before launch")
    out = ROOT / "results/b200_vllm031" / name
    out.mkdir(parents=True, exist_ok=False)
    settings = json.loads((EXPERIMENT / "config.json").read_text())
    parent_command = parent["command"]
    condition = json.loads(
        parent_command[parent_command.index("--additional-config") + 1]
    )["serving_condition"].copy()
    condition["mxfp8_validation"] = native_receipt
    condition["fp4_prepare_validation"]["vendor"] = vendor_receipt
    archive = ROOT / "results/b200_vllm031/pre_migration_sources.tar.gz"
    restored = restore_diagnostic_sources(ROOT, condition, archive)
    write(
        out / "source_restoration.json",
        {
            "archive_sha256": sha(archive),
            "restored": restored,
            "runtime_sources_replaced": False,
        },
    )
    command = candidate_command(
        parent["command"],
        native_receipt,
        vendor_receipt,
        mode,
        gdn_decode_kernel,
        gdn_cp,
        enforce_eager,
        max_active,
    )
    env = candidate_environment(ROOT)
    env["VLLM_GDN_DECODE_KERNEL"] = gdn_decode_kernel
    env["GLEIPNIR_FLASHINFER_GDN_CP"] = gdn_cp
    if shutil.which("ninja", path=env["PATH"]) is None:
        raise ValueError("FlashInfer CUDA JIT requires ninja on the candidate PATH")
    env["GLEIPNIR_GIGATOKEN_RECEIPT"] = str(out / "frontend.json")
    env["GLEIPNIR_FROST_WRAPPER_VALIDATION"] = parent["host_wrapper"]["validation"]
    from vllm.v1.core.sched import scheduler

    scheduler_binding = {
        "upstream_sha256": sha(Path(inspect.getfile(scheduler))),
        "integration_sha256": sha(EXPERIMENT / "scheduler.py"),
    }
    env["GLEIPNIR_VLLM031_SCHEDULER_BINDING"] = json.dumps(scheduler_binding)
    manifest = json.loads((DATA / "manifest.json").read_text())
    workloads = {}
    for key, digest in manifest["files"].items():
        if sha(DATA / f"{key}.json") != digest:
            raise ValueError("frozen prompt drift")
        workloads[key] = json.loads((DATA / f"{key}.json").read_text())
    source_paths = sorted(
        {
            *[str(p.relative_to(ROOT)) for p in EXPERIMENT.glob("*.py")],
            "src/gleipnir/serving/mxfp8.py",
            "src/gleipnir/serving/monitor_score.py",
            "src/gleipnir/serving/compile_cache.py",
            "experiments/b200_attention_gdn_serving/mxfp8_worker.py",
            "experiments/b200_attention_gdn_serving/worker.py",
        }
    )
    source_hashes = {p: sha(ROOT / p) for p in source_paths}
    for p in source_paths:
        target = out / "executed_sources" / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / p).read_bytes())
    report = {
        "status": "starting",
        "promoted": False,
        "settings": settings,
        "scheduler_mode": mode,
        "gdn_decode_kernel": gdn_decode_kernel,
        "gdn_cp": gdn_cp,
        "enforce_eager": enforce_eager,
        "finite_canary_diagnostic": allow_diagnostic,
        "max_num_active_seqs": max_active,
        "gdn_receipt_sha256": sha(ROOT / gdn_receipt) if gdn_receipt else None,
        "scheduler_binding": scheduler_binding,
        "command": command,
        "source_hashes": source_hashes,
        "runtime": {
            n: importlib.metadata.version(n)
            for n in ("vllm", "torch", "triton", "transformers", "flashinfer-python")
        },
        "analysis_runtime": {
            n: importlib.metadata.version(n)
            for n in ("numpy", "pandas", "scikit-learn", "scipy")
        },
        "native_receipt_sha256": sha(ROOT / native_receipt),
        "vendor_receipt_sha256": sha(ROOT / vendor_receipt),
        "hardware": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "trials": [],
    }
    write(out / "reference_selection.json", selection)
    write(out / "manifest.json", manifest)
    write(out / "parent_server.json", parent)
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
    report["pid"] = process.pid
    report["server_log"] = str(log.relative_to(ROOT))
    write(out / "summary.json", report)
    active = {
        "pid": process.pid,
        "command": command,
        "status": "starting",
        "started_at_unix": time.time(),
        "runner": "pooling",
        "endpoint": settings["endpoint"],
        "log": str(log),
        "frontend": {**parent["frontend"], "receipt_path": str(out / "frontend.json")},
        "host_wrapper": parent["host_wrapper"],
        "cache_paths": {
            key: value for key, value in env.items() if key in parent["cache_paths"]
        },
        "runtime_migration": report["runtime"],
        "migration_results": str(out.relative_to(ROOT)),
        "score_sources": source_hashes,
    }
    write(active_path, active)
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{settings['port']}", trust_env=False, timeout=30
        ) as client:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"candidate server exited: {process.returncode}")
                try:
                    if (await client.get("/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.perf_counter() - started > 1200:
                    raise TimeoutError("candidate startup exceeded twenty minutes")
                await asyncio.sleep(2)
        report.update(status="running", ready_seconds=time.perf_counter() - started)
        active.update(status="ready", ready_at_unix=time.time())
        write(active_path, active)
        write(out / "summary.json", report)
        print("candidate_ready", report["ready_seconds"], flush=True)
        controls = json.loads((ROOT / settings["canary_reference"]).read_text())
        values, _ = await trial(workloads["canary"], 4, settings)
        if len(values) != 20:
            raise ValueError("twenty-row score canary is incomplete")
        scores = np.array([v["score"] for v in values])
        reference = np.array(controls["served"]["adapter"])
        mean = float(np.mean(np.abs(scores - reference)))
        correlation = float(np.corrcoef(scores, reference)[0, 1])
        effect = float(np.max(np.abs(scores - np.array(controls["served"]["base"]))))
        canary = {
            "mean_absolute_difference": mean,
            "correlation": correlation,
            "adapter_effect": effect,
            "inherited_strict_master_passed": controls["passed"],
            "reference_sha256": sha(ROOT / settings["canary_reference"]),
            "passed": mean <= settings["canary_mean_error_limit"]
            and correlation >= settings["canary_correlation_floor"]
            and effect > 0,
            "finite": bool(np.isfinite(scores).all())
            and all(np.isfinite(value["margin"]) for value in values),
            "diagnostic_continuation_authorized": allow_diagnostic,
        }
        frozen_config = yaml.safe_load(
            (ROOT / "experiments/b200_inference_benchmark/config.yaml").read_text()
        )
        master_path = ROOT / frozen_config["parity"]
        if sha(master_path) != frozen_config["parity_sha256"]:
            raise ValueError("frozen master parity reference changed")
        master = json.loads(master_path.read_text())
        if master["reference"]["prompt_sha256"] != [
            row["prompt_sha256"] for row in workloads["canary"]
        ]:
            raise ValueError("master score prompt identities changed")
        master_scores = np.array(master["reference"]["adapter"])
        master_mean = float(np.mean(np.abs(scores - master_scores)))
        master_correlation = float(np.corrcoef(scores, master_scores)[0, 1])
        canary["strict_master"] = {
            "reference_sha256": sha(master_path),
            "mean_absolute_difference": master_mean,
            "correlation": master_correlation,
            "passed": master_mean <= master["limits"]["max_mean_absolute_difference"]
            and master_correlation >= master["limits"]["min_correlation"],
        }
        write(out / "canary.json", canary)
        write(out / "canary_predictions.json", values)
        archive_audits(out)
        report["canary_passed"] = canary["passed"]
        if not may_benchmark(canary, allow_diagnostic):
            raise ValueError("candidate score canary failed")
        print("candidate_canary", canary, flush=True)
        baseline = ROOT / selection["results"]
        for concurrency, key, repeats in ((1, "quick", 3), (128, "full", 6)):
            warmup, _ = await trial(workloads[key], concurrency, settings)
            write(out / f"c{concurrency}_warmup.json", warmup)
            candidate = []
            for repeat in range(repeats):
                values, seconds = await trial(workloads[key], concurrency, settings)
                candidate.append(values)
                write(out / f"c{concurrency}_repeat{repeat}.json", values)
                report["trials"].append(
                    {
                        "concurrency": concurrency,
                        "repeat": repeat,
                        **measurement_summary(values, seconds),
                    }
                )
                write(out / "summary.json", report)
                print("candidate_pass", concurrency, repeat, seconds, flush=True)
            archived = [
                json.loads(p.read_text())
                for p in sorted(baseline.glob(f"c{concurrency}_repeat*.json"))
            ]
            write(
                out / f"c{concurrency}_comparison.json",
                {
                    "scores": paired_score_summary(archived, candidate),
                    "ranking": ranking_comparison(workloads[key], archived, candidate),
                },
            )
        archive_audits(out)
        report["status"] = "complete"
        write(out / "summary.json", report)
    except BaseException as error:
        if report.get("ready_seconds") is not None:
            archive_audits(out)
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write(out / "summary.json", report)
        if process.poll() is None:
            import os
            import signal

            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=60)
        active["status"] = "failed"
        write(out / "failed_server.json", active)
        active_path.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--native-receipt", required=True)
    parser.add_argument("--vendor-receipt", required=True)
    parser.add_argument(
        "--scheduler-mode", choices=("default", "adaptive"), default="default"
    )
    parser.add_argument(
        "--gdn-decode-kernel", choices=("cuda", "triton"), default="cuda"
    )
    parser.add_argument("--gdn-cp", choices=("auto", "off"), default="off")
    parser.add_argument("--gdn-receipt")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-num-active-seqs", type=int)
    parser.add_argument(
        "--finite-canary-diagnostic",
        action="store_true",
        help="Requires explicit user authorization to time a finite failed canary",
    )
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {"", ".", ".."}:
        raise ValueError("run name must be a stem")
    asyncio.run(
        run(
            args.name,
            args.native_receipt,
            args.vendor_receipt,
            args.scheduler_mode,
            args.gdn_decode_kernel,
            args.gdn_cp,
            args.gdn_receipt,
            args.enforce_eager,
            args.finite_canary_diagnostic,
            args.max_num_active_seqs,
        )
    )


if __name__ == "__main__":
    main()
