"""Launch the selected monitor backbone in explicit eager Lens research mode."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

from gleipnir.serving.reference import selected_serving_default

ROOT = Path(__file__).resolve().parents[2]
SERVING = ROOT / "results/b200_attention_gdn_serving"
OVERLAY = Path("/tmp/gleipnir-vllm-lens-1.3.0")


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


async def start(out: Path) -> dict:
    from experiments.b200_long_context.run import gpu
    from experiments.b200_vllm031.runtime import candidate_environment
    from experiments.b200_vllm031.source_bindings import restore_diagnostic_sources

    if gpu()["apps"] or (SERVING / "server.json").exists():
        raise ValueError("retire the identity-verified previous scorer first")
    selection, command = selected_serving_default(ROOT)
    if {p: importlib.metadata.version(p) for p in selection["runtime"]} != selection[
        "runtime"
    ]:
        raise ValueError("selected serving packages changed")
    stage = json.loads((ROOT / "results/b200_vllm_lens/staging.json").read_text())
    for name, sha in stage["files_sha256"].items():
        if hashlib.sha256((OVERLAY / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Lens overlay drift: {name}")
    archive = ROOT / "results/b200_vllm031/pre_migration_sources.tar.gz"
    previous = json.loads(
        (
            ROOT / "results/b200_vllm031/diagnostic_default02/source_restoration.json"
        ).read_text()
    )
    if hashlib.sha256(archive.read_bytes()).hexdigest() != previous["archive_sha256"]:
        raise ValueError("diagnostic source archive drift")
    condition = json.loads(command[command.index("--additional-config") + 1])[
        "serving_condition"
    ]
    restored = restore_diagnostic_sources(ROOT, condition, archive)
    write(out / "source_restoration.json", restored)
    merged = Path(command[command.index("--model") + 1])
    manifest = json.loads((ROOT / selection["merged_artifact"]).read_text())
    if json.loads((merged / "merge_manifest.json").read_text()) != manifest:
        raise ValueError("merged adapter provenance changed")
    for name, sha in manifest["files_sha256"].items():
        if hashlib.sha256((merged / name).read_bytes()).hexdigest() != sha:
            raise ValueError("merged adapter weight drift")
    command[0] = sys.executable
    command[2] = "experiments.b200_vllm_lens.server"
    command += [
        "--enforce-eager",
        "--worker-extension-cls",
        "gleipnir.serving.lens_worker.MonitorLensExtension",
    ]
    env = candidate_environment(ROOT)
    env["PYTHONPATH"] += ":" + str(OVERLAY)
    env["VLLM_LENS_DISABLE"] = "1"
    env["VLLM_GDN_DECODE_KERNEL"] = "cuda"
    env["GLEIPNIR_FLASHINFER_GDN_CP"] = "auto"
    parent = json.loads((ROOT / selection["host_parent"]).read_text())
    env["GLEIPNIR_FROST_WRAPPER_VALIDATION"] = parent["host_wrapper"]["validation"]
    env["GLEIPNIR_GIGATOKEN_RECEIPT"] = str(out / "frontend.json")
    env["GLEIPNIR_VLLM031_SCHEDULER_BINDING"] = json.dumps(
        selection["scheduler_binding"]
    )
    log = ROOT / "logs/runpod/b200_vllm_lens" / f"{out.name}_server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    before = time.perf_counter()
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
        "log": str(log),
        "runtime_migration": selection["runtime"],
        "serving_default": selection["name"],
        "lens_research": True,
        "enforce_eager": True,
        "lens_stage": stage,
        "frontend": parent["frontend"] | {"receipt_path": str(out / "frontend.json")},
        "host_wrapper": parent["host_wrapper"],
    }
    write(SERVING / "server.json", active)
    write(out / "server.json", active)
    write(out / "selection.json", selection)
    try:
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8010", trust_env=False, timeout=300
        ) as client:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Lens server exited: {process.returncode}")
                try:
                    if (await client.get("/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.perf_counter() - before > 1200:
                    raise TimeoutError("Lens startup exceeded twenty minutes")
                await asyncio.sleep(2)
            info = await client.get("/v1/monitor/lens/info")
            info.raise_for_status()
            shape = info.json()
            if shape["layers"] != list(range(32)) or shape["hidden_size"] != 2560:
                raise ValueError("monitor Lens model envelope changed")
            write(out / "lens_info.json", shape)
        active.update(status="ready", ready_at_unix=time.time())
        write(SERVING / "server.json", active)
        write(out / "server.json", active)
        write(
            out / "startup.json",
            {"seconds": time.perf_counter() - before, "passed": True},
        )
        print("monitor_lens_ready", process.pid, shape, flush=True)
        return shape
    except BaseException:
        if process.poll() is None:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.b200_vllm031.stop",
                    "--archive-name",
                    out.name + "_failed",
                ],
                env=env,
                cwd=ROOT,
                check=True,
            )
        elif not gpu()["apps"]:
            (SERVING / "server.json").unlink(missing_ok=True)
        raise
