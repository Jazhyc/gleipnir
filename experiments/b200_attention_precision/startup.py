"""Prepare or launch the user-selected FP8 projection default without a sweep."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from experiments.b200_inference_benchmark.run import DATA, ROOT, sha, write
from experiments.b200_long_context.run import gpu, rpc
from experiments.b200_monitor_score.run import SERVING, trial
from experiments.b200_vllm031.run import archive_audits
from experiments.b200_vllm031.runtime import candidate_environment
from experiments.b200_vllm031.source_bindings import restore_diagnostic_sources
from gleipnir.serving.reference import selected_serving_default


async def startup(name: str, *, prepare_only: bool = False) -> None:
    """Reproduce the accepted recipe and guard its canary before retaining it."""
    if not name or Path(name).name != name:
        raise ValueError("startup name must be a stem")
    selection, command = selected_serving_default(ROOT)
    if {n: importlib.metadata.version(n) for n in selection["runtime"]} != selection[
        "runtime"
    ]:
        raise ValueError("accepted serving runtime changed")
    from vllm.v1.core.sched import scheduler

    binding = {
        "upstream_sha256": sha(Path(inspect.getfile(scheduler))),
        "integration_sha256": sha(ROOT / "experiments/b200_vllm031/scheduler.py"),
    }
    if binding != selection["scheduler_binding"]:
        raise ValueError("accepted pooling scheduler source changed")
    command[0] = sys.executable
    out = ROOT / "results/b200_attention_precision" / name
    out.mkdir(parents=True, exist_ok=False)
    archive = ROOT / "results/b200_vllm031/pre_migration_sources.tar.gz"
    archive_receipt = json.loads(
        (
            ROOT / "results/b200_vllm031/diagnostic_default02/source_restoration.json"
        ).read_text()
    )
    if sha(archive) != archive_receipt["archive_sha256"]:
        raise ValueError("diagnostic source archive checksum changed")
    additional = json.loads(command[command.index("--additional-config") + 1])
    restored = restore_diagnostic_sources(
        ROOT, additional["serving_condition"], archive
    )
    write(
        out / "source_restoration.json",
        {
            "archive_sha256": sha(archive),
            "restored": restored,
            "runtime_sources_replaced": False,
        },
    )
    write(out / "selection.json", selection)
    write(out / "command.json", command)
    write(out / "startup_source.json", {"sha256": sha(Path(__file__))})
    if prepare_only:
        write(out / "summary.json", {"status": "prepared", "gpu_started": False})
        print("fp8_default_prepared", selection["name"], flush=True)
        return
    snapshot = gpu()
    if snapshot["apps"] or (SERVING / "server.json").exists():
        raise ValueError("retire the existing server before starting the default")
    model = Path(command[command.index("--model") + 1])
    merge = json.loads((ROOT / selection["merged_artifact"]).read_text())
    if json.loads((model / "merge_manifest.json").read_text()) != merge:
        raise ValueError("accepted merged adapter provenance changed")
    for path, expected in merge["files_sha256"].items():
        if sha(model / path) != expected:
            raise ValueError(f"accepted merged weight changed: {path}")
    env = candidate_environment(ROOT)
    parent = json.loads((ROOT / selection["host_parent"]).read_text())
    env["GLEIPNIR_FROST_WRAPPER_VALIDATION"] = parent["host_wrapper"]["validation"]
    env["GLEIPNIR_GIGATOKEN_RECEIPT"] = str(out / "frontend.json")
    env["VLLM_GDN_DECODE_KERNEL"] = "cuda"
    env["GLEIPNIR_FLASHINFER_GDN_CP"] = "auto"
    env["GLEIPNIR_VLLM031_SCHEDULER_BINDING"] = json.dumps(binding)
    log = ROOT / "logs/runpod/b200_attention_precision" / f"{name}_server.log"
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
        "runtime_migration": selection["runtime"],
        "log": str(log),
        "serving_default": selection["name"],
        "frontend": parent["frontend"] | {"receipt_path": str(out / "frontend.json")},
        "host_wrapper": parent["host_wrapper"],
    }
    write(SERVING / "server.json", active)
    write(out / "server.json", active)
    try:
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8010", trust_env=False, timeout=30
        ) as client:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"default server exited: {process.returncode}")
                try:
                    if (await client.get("/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.perf_counter() - started > 1200:
                    raise TimeoutError("default startup exceeded twenty minutes")
                await asyncio.sleep(2)
        if (await rpc("frost_wrapper_state"))["mode"] != "direct":
            raise ValueError("default direct FROST binding changed")
        settings = {"port": 8010, "timeout_seconds": 300}
        values, _ = await trial(
            json.loads((DATA / "canary.json").read_text()), 4, settings
        )
        previous = json.loads((ROOT / selection["canary_predictions"]).read_text())
        if [(r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in values] != [
            (r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in previous
        ]:
            raise ValueError("default canary identity changed")
        scores = np.array([r["score"] for r in values])
        expected = np.array([r["score"] for r in previous])
        mae = float(np.abs(scores - expected).mean())
        correlation = float(np.corrcoef(scores, expected)[0, 1])
        master = json.loads((ROOT / selection["master_canary"]).read_text())
        if [r["prompt_sha256"] for r in values] != master["prompt_sha256"]:
            raise ValueError("default master canary prompt identity changed")
        effect = float(np.abs(scores - np.array(master["base"])).max())
        limits = selection["reproduction_limits"]
        receipt = {
            "passed": bool(np.isfinite(scores).all())
            and mae <= limits["score_mae"]
            and correlation >= limits["correlation"]
            and bool(np.isfinite(effect) and effect > 0),
            "adapter_effect": effect,
            "score_mae": mae,
            "correlation": correlation,
            "quality_status": "user_accepted_finite",
            "inherited_strict_master_passed": False,
            "inherited_strict_old_serving_passed": False,
        }
        write(out / "canary.json", receipt)
        write(out / "canary_predictions.json", values)
        archive_audits(out)
        projection_audit = json.loads(
            (out / "native_attention_projections.json").read_text()
        )
        if (
            not projection_audit["passed"]
            or projection_audit["precision"] != "fp8"
            or len(projection_audit["calls"]) != 16
            or projection_audit["validation_sha256"]
            != sha(ROOT / selection["native_validation"])
        ):
            raise ValueError("default actual FP8 projection dispatch changed")
        if not receipt["passed"]:
            raise ValueError("accepted FP8 default reproduction failed")
        active.update(status="ready", ready_at_unix=time.time())
        write(SERVING / "server.json", active)
        write(out / "server.json", active)
        write(
            out / "summary.json",
            {
                "status": "ready",
                "startup_seconds": time.perf_counter() - started,
                "quality_status": "user_accepted_finite",
                "gpu_started": True,
            },
        )
        print("fp8_default_ready", process.pid, mae, correlation, flush=True)
    except BaseException as error:
        write(out / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        if process.poll() is None:
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
        elif not gpu()["apps"] and (SERVING / "server.json").exists():
            if json.loads((SERVING / "server.json").read_text())["pid"] == process.pid:
                (SERVING / "server.json").unlink()
        write(out / "closure.json", gpu())
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(startup(args.name, prepare_only=args.prepare_only))


if __name__ == "__main__":
    main()
