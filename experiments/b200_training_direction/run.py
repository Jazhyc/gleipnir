"""Capture matched training views, sequentially replacing base/merged engines."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from gleipnir.data.monitoring import file_hash, read_rows, write_json
from gleipnir.evaluation.lens_capture import capture_rows

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
CONFIG = HERE / "config.json"
ACTIVE = ROOT / "results/b200_attention_gdn_serving/server.json"


def sources() -> dict[str, str]:
    paths = list(HERE.glob("*.py")) + [HERE / "README.md"]
    paths += list((ROOT / "src/gleipnir/serving").glob("lens*.py"))
    paths += [
        ROOT / "src/gleipnir/evaluation/lens_capture.py",
        ROOT / "src/gleipnir/serving/bf16_worker.py",
    ]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def checked(c: dict, out: Path) -> dict:
    m = json.loads((out / "manifest.json").read_text())
    if file_hash(CONFIG) != m["config_sha256"] or sources() != m["sources"]:
        raise ValueError("source/config drift")
    for spec in c["inputs"].values():
        if file_hash(ROOT / spec["path"]) != spec["sha256"]:
            raise ValueError("input drift " + spec["path"])
    if file_hash(out / "workload.jsonl") != m["workload_sha256"]:
        raise ValueError("workload drift")
    server = json.loads(ACTIVE.read_text())
    actual = [
        v.decode()
        for v in Path(f"/proc/{server['pid']}/cmdline").read_bytes().split(b"\0")
        if v
    ]
    if actual != server["command"] or os.getpgid(server["pid"]) != server["pid"]:
        raise ValueError("resident process drift")
    return server


async def info() -> dict:
    async with httpx.AsyncClient(trust_env=False, timeout=300) as client:
        r = await client.get("http://127.0.0.1:8010/v1/monitor/lens/info")
        r.raise_for_status()
        v = r.json()
    if (
        v["layers"] != list(range(32))
        or v["hidden_size"] != 2560
        or any(
            v[k]
            for k in ["capture_requests", "steering_requests", "projection_requests"]
        )
    ):
        raise ValueError("Lens geometry or cleanup failed")
    return v


def replacement_command(parent: dict, model: str) -> list[str]:
    """Change only model identity in the matched research command."""
    command = list(parent["command"])
    command[command.index("--model") + 1] = model
    i = command.index("--additional-config") + 1
    additional = json.loads(command[i])
    additional["serving_condition"]["merged_model"] = model
    command[i] = json.dumps(additional, sort_keys=True)
    return command


async def replace(c: dict, out: Path, parent: dict, name: str, model: str) -> dict:
    from experiments.b200_vllm031.runtime import candidate_environment

    checked(c, out)
    retired = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.b200_vllm031.stop",
            "--archive-name",
            c["campaign_id"] + "_before_" + name,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    print(retired.stdout.strip(), flush=True)
    command = replacement_command(parent, model)
    env = candidate_environment(ROOT)
    env["PYTHONPATH"] += ":/tmp/gleipnir-vllm-lens-1.3.0:/tmp/gleipnir-gigatoken-0.10.0"
    extra = json.loads(command[command.index("--additional-config") + 1])
    # Reuse the corrected scheduler binding from the passing parent environment.
    import inspect

    from vllm.v1.core.sched import scheduler

    env.update(
        GLEIPNIR_VLLM031_SCHEDULER_BINDING=json.dumps(
            {
                "upstream_sha256": file_hash(Path(inspect.getfile(scheduler))),
                "integration_sha256": file_hash(
                    ROOT / "experiments/b200_vllm031/scheduler.py"
                ),
            }
        ),
        VLLM_LENS_DISABLE="1",
        VLLM_GDN_DECODE_KERNEL="cuda",
        GLEIPNIR_FLASHINFER_GDN_CP="auto",
        GLEIPNIR_BF16_AUDIT=str(out / (name + "_bf16_audit.json")),
        GLEIPNIR_GIGATOKEN_RECEIPT=str(out / (name + "_frontend.json")),
    )
    if extra["serving_condition"]["quantization"] is not None:
        raise ValueError("quantized replacement forbidden")
    env.pop("GLEIPNIR_FROST_WRAPPER_VALIDATION", None)
    log = ROOT / "logs/runpod/b200_training_direction" / (name + "_server.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as f:
        p = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    server = parent | {
        "pid": p.pid,
        "command": command,
        "log": str(log),
        "status": "starting",
        "model_role": name,
        "launched_at_unix": time.time(),
    }
    if name == "base":
        server.pop("adapter_sha256", None)
    write_json(ACTIVE, server)
    write_json(out / (name + "_server.json"), server)
    started = time.perf_counter()
    async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
        while True:
            if p.poll() is not None:
                raise RuntimeError(f"replacement {name} exited {p.returncode}")
            try:
                if (
                    await client.get("http://127.0.0.1:8010/health")
                ).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.perf_counter() - started > 1200:
                raise TimeoutError("replacement startup exceeded 20 minutes")
            await asyncio.sleep(2)
    server.update(status="ready", ready_at_unix=time.time())
    write_json(ACTIVE, server)
    write_json(out / (name + "_server.json"), server)
    print("training_direction_engine_ready", name, p.pid, flush=True)
    await info()
    return server


async def run() -> None:
    c = json.loads(CONFIG.read_text())
    out = ROOT / "results/b200_training_direction" / c["campaign_id"]
    hardware = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,ecc.errors.uncorrected.volatile.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if (
        hardware.split(",")[0] != c["gpu_uuid"]
        or hardware.split(",")[-1].strip() != "0"
    ):
        raise ValueError("GPU identity/health drift")
    write_json(out / "hardware.json", {"gpu": hardware})
    parent = json.loads((ROOT / c["inputs"]["parent_manifest"]["path"]).read_text())[
        "server"
    ]
    for filename, sha in c["base_files_sha256"].items():
        if file_hash(ROOT / c["base_model"] / filename) != sha:
            raise ValueError("base checkpoint drift")
    merged = json.loads((ROOT / c["inputs"]["merged_artifact"]["path"]).read_text())
    for filename, sha in merged["files_sha256"].items():
        if file_hash(Path(c["trained_model"]) / filename) != sha:
            raise ValueError("merged checkpoint drift")
    if {k: importlib.metadata.version(k) for k in c["runtime"]} != c["runtime"]:
        raise ValueError("runtime drift")
    if json.loads(ACTIVE.read_text()) != parent:
        raise ValueError("requires original passing trained engine")
    if not json.loads((ROOT / c["inputs"]["merge_parity"]["path"]).read_text())[
        "passed"
    ]:
        raise ValueError("merge parity failed")
    m = {
        "sources": sources(),
        "config_sha256": file_hash(CONFIG),
        "workload_sha256": file_hash(out / "workload.jsonl"),
        "server": parent,
        "inputs": c["inputs"],
        "base_files_sha256": c["base_files_sha256"],
    }
    for path in m["sources"]:
        target = out / "executed_sources" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    write_json(out / "manifest.json", m)
    shutil.copyfile(CONFIG, out / "config.json")
    checked(c, out)
    unit = np.load(ROOT / c["inputs"]["directions"]["path"])["unit"]
    workload = read_rows(out / "workload.jsonl")
    canary = read_rows(ROOT / c["inputs"]["canary"]["path"])
    reference = json.loads((ROOT / c["inputs"]["master_reference"]["path"]).read_text())
    previous = json.loads((ROOT / c["inputs"]["previous_scores"]["path"]).read_text())
    if [r["prompt_sha256"] for r in canary] != reference["prompt_sha256"]:
        raise ValueError("master reference prompt mismatch")
    if [r["prompt_sha256"] for r in canary] != [r["prompt_sha256"] for r in previous]:
        raise ValueError("eager reference prompt mismatch")

    async def capture(rows, name):
        return await capture_rows(
            rows,
            out,
            name,
            surface="01",
            unit=unit,
            concurrency=c["concurrency"],
            batch_rows=c["batch_rows"],
            timeout_seconds=c["timeout_seconds"],
        )

    async def gate(name, role):
        checked(c, out)
        rs = await capture(canary, name + "_canary")
        x = np.array([r["score"] for r in rs])
        y = np.array(reference[role])
        mae = float(abs(x - y).mean())
        corr = float(np.corrcoef(x, y)[0, 1])
        receipt = {
            "mae": mae,
            "correlation": corr,
            "role": role,
            "passed": mae <= 0.020 and corr >= 0.99 and np.isfinite(x).all(),
        }
        if role == "adapter":
            old = np.array([r["score"] for r in previous])
            receipt.update(
                eager_mae=float(abs(x - old).mean()),
                eager_correlation=float(np.corrcoef(x, old)[0, 1]),
                adapter_effect=float(abs(x - np.array(reference["base"])).mean()),
            )
            receipt["passed"] = bool(
                receipt["passed"]
                and receipt["eager_mae"] <= 0.005
                and receipt["eager_correlation"] >= 0.995
                and receipt["adapter_effect"] >= 1e-6
            )
        receipt["passed"] = bool(receipt["passed"])
        write_json(out / (name + "_gate.json"), receipt)
        if not receipt["passed"]:
            raise ValueError("master/eager canary failed " + name)
        audit = json.loads((ACTIVE.parent / "loaded_precision.json").read_text())
        if not audit["passed"] or len(audit["attention_calls"]) != 8:
            raise ValueError("native BF16 dispatch incomplete")
        write_json(out / (name + "_native_audit.json"), audit)
        write_json(out / (name + "_info.json"), await info())
        print("training_direction_gate_passed", name, mae, corr, flush=True)

    await gate("trained", "adapter")
    await capture(workload, "trained")
    checked(c, out)
    write_json(out / "trained_closure.json", await info())
    await replace(c, out, parent, "base", str(ROOT / c["base_model"]))
    await gate("base", "base")
    await capture(workload, "base")
    checked(c, out)
    write_json(out / "base_closure.json", await info())
    await replace(c, out, parent, "restored", c["trained_model"])
    await gate("restored", "adapter")
    from experiments.b200_training_direction.analyze import analyze

    analyze(out, ROOT, c)
    write_json(out / "closure_info.json", await info())
    write_json(out / "status.json", {"stage": "complete"})
    print("training_direction_complete", flush=True)


def main() -> None:
    sys.path.insert(0, "/tmp/gleipnir-vllm-lens-1.3.0")
    c = json.loads(CONFIG.read_text())
    out = ROOT / "results/b200_training_direction" / c["campaign_id"]
    try:
        asyncio.run(run())
    except BaseException as e:
        write_json(out / "failure.json", {"type": type(e).__name__, "message": str(e)})
        raise


if __name__ == "__main__":
    main()
