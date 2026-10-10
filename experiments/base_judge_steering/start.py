"""Replace the warm adapted Lens process with the pinned BF16 base model."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from experiments.b200_vllm031.runtime import candidate_environment
from gleipnir.data.monitoring import file_hash, write_json
from gleipnir.evaluation.direction_campaign import info, replacement_command

ROOT = Path(__file__).resolve().parents[2]
ACTIVE = ROOT / "results/b200_attention_gdn_serving/server.json"


async def start(config: dict, out: Path, parent: dict) -> dict:
    """Reuse the matched replacement command, caches and native overlays."""
    from vllm.v1.core.sched import scheduler

    active = json.loads(ACTIVE.read_text())
    actual = [
        v.decode()
        for v in Path(f"/proc/{parent['pid']}/cmdline").read_bytes().split(b"\0")
        if v
    ]
    if (
        active != parent
        or actual != parent["command"]
        or os.getpgid(parent["pid"]) != parent["pid"]
    ):
        raise ValueError("resident parent identity drift")
    if (
        parent["serving_precision"] != "bf16"
        or not parent["lens_research"]
        or "--enforce-eager" not in actual
        or any(flag in actual for flag in ("--enable-lora", "--lora-modules"))
    ):
        raise ValueError("base requires eager BF16 without dynamic adapters")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.b200_vllm031.stop",
            "--archive-name",
            "base_judge_steering_before_base01",
        ],
        cwd=ROOT,
        check=True,
    )
    command = replacement_command(parent, str(ROOT / config["base_model"]))
    env = candidate_environment(ROOT)
    env["PYTHONPATH"] += ":/tmp/gleipnir-vllm-lens-1.3.0:/tmp/gleipnir-gigatoken-0.10.0"
    env.update(
        VLLM_LENS_DISABLE="1",
        VLLM_GDN_DECODE_KERNEL="cuda",
        GLEIPNIR_FLASHINFER_GDN_CP="auto",
        GLEIPNIR_BF16_AUDIT=str(out / "base_bf16_audit.json"),
        GLEIPNIR_GIGATOKEN_RECEIPT=str(out / "base_frontend.json"),
        GLEIPNIR_VLLM031_SCHEDULER_BINDING=json.dumps(
            {
                "upstream_sha256": file_hash(Path(inspect.getfile(scheduler))),
                "integration_sha256": file_hash(
                    ROOT / "experiments/b200_vllm031/scheduler.py"
                ),
            }
        ),
    )
    env.pop("GLEIPNIR_FROST_WRAPPER_VALIDATION", None)
    log = ROOT / "logs/runpod/base_judge_steering/base_server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    server = parent | {
        "pid": process.pid,
        "command": command,
        "status": "starting",
        "log": str(log),
        "model_role": "base",
        "launched_at_unix": time.time(),
        "frontend": parent["frontend"]
        | {"receipt_path": str(out / "base_frontend.json"), "ab_control": False},
    }
    server.pop("adapter_sha256", None)
    server.pop("ready_at_unix", None)
    write_json(ACTIVE, server)
    write_json(out / "server.json", server)
    try:
        started = time.perf_counter()
        async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"base server exited {process.returncode}")
                try:
                    if (
                        await client.get("http://127.0.0.1:8010/health")
                    ).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.perf_counter() - started > 1200:
                    raise TimeoutError("base Lens startup exceeded 20 minutes")
                await asyncio.sleep(2)
        server.update(status="ready", ready_at_unix=time.time())
        write_json(ACTIVE, server)
        write_json(out / "server.json", server)
        await info()
        print("base_judge_engine_ready", process.pid, flush=True)
        return server
    except BaseException:
        if process.poll() is None:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.b200_vllm031.stop",
                    "--archive-name",
                    "base_judge_start_failed01",
                ],
                cwd=ROOT,
                check=True,
                env=env,
            )
        raise
