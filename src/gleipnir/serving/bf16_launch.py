"""Launch the BF16 monitor server for any merged Qwen3.5-family model.

Mirrors the campaign's BF16 evaluation launch (selected serving command, the
merged model, ``bf16_monitor.command``, the audited runtime environment and the
source-bound scheduler) so systems experiments start the same server. A
candidate changes only the worker class and may add ``--additional-config``
keys; everything else, including the model's own text config, is derived.
"""

from __future__ import annotations

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

from gleipnir.data.monitoring import file_hash, write_json

BF16_WORKER = "gleipnir.serving.bf16_worker.Bf16Worker"


def bf16_command(
    root: Path,
    merged_model: Path,
    sources: dict[str, str],
    *,
    worker: str = BF16_WORKER,
    additional: dict | None = None,
) -> tuple[dict, list[str]]:
    """Return the selection and BF16 server command for ``merged_model``."""
    from gleipnir.serving.bf16_monitor import command as convert
    from gleipnir.serving.reference import selected_serving_default

    selection, command = selected_serving_default(root, last_query=False)
    if {n: importlib.metadata.version(n) for n in selection["runtime"]} != selection[
        "runtime"
    ]:
        raise ValueError("selected serving runtime changed")
    command[0] = sys.executable
    command[command.index("--model") + 1] = str(merged_model)
    command = convert(command, sources)
    command[command.index("--worker-cls") + 1] = worker
    i = command.index("--additional-config") + 1
    config = json.loads(command[i])
    config["serving_condition"]["merged_model"] = str(merged_model)
    config.update(additional or {})
    command[i] = json.dumps(config, sort_keys=True)
    return selection, command


def environment(root: Path, selection: dict, audit: Path) -> dict[str, str]:
    """The campaign's BF16 serving environment, with a scheduler binding check."""
    from vllm.v1.core.sched import scheduler

    from experiments.b200_vllm031.runtime import candidate_environment

    binding = {
        "upstream_sha256": file_hash(Path(inspect.getfile(scheduler))),
        "integration_sha256": file_hash(root / "experiments/b200_vllm031/scheduler.py"),
    }
    if binding != selection["scheduler_binding"]:
        raise ValueError("selected pooling scheduler changed")
    env = candidate_environment(root)
    parent = json.loads((root / selection["host_parent"]).read_text())
    env.update(
        GLEIPNIR_GIGATOKEN_RECEIPT=str(audit.parent / "frontend.json"),
        VLLM_GDN_DECODE_KERNEL="cuda",
        GLEIPNIR_FLASHINFER_GDN_CP="auto",
        GLEIPNIR_VLLM031_SCHEDULER_BINDING=json.dumps(binding),
        GLEIPNIR_BF16_AUDIT=str(audit),
    )
    env.pop("GLEIPNIR_FROST_WRAPPER_VALIDATION", None)
    if parent.get("frontend") is None:
        raise ValueError("selected host parent has no frontend binding")
    return env


def start(
    root: Path, command: list[str], env: dict, log: Path, receipt: Path
) -> subprocess.Popen:
    """Start the server in its own session and record its identity."""
    from experiments.b200_vllm031.source_bindings import restore_diagnostic_sources

    pids = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).split()
    if pids:
        raise ValueError(f"GPU is busy: {pids}")
    condition = json.loads(command[command.index("--additional-config") + 1])
    restore_diagnostic_sources(
        root,
        condition["serving_condition"],
        root / "results/b200_vllm031/pre_migration_sources.tar.gz",
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as handle:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write_json(
        receipt,
        {"pid": process.pid, "command": command, "log": str(log), "status": "starting"},
    )
    return process


def wait_ready(process: subprocess.Popen, port: int, timeout: float = 1800) -> float:
    """Wait for /health; returns startup seconds."""
    started = time.perf_counter()
    with httpx.Client(
        base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=30
    ) as client:
        while time.perf_counter() - started < timeout:
            if process.poll() is not None:
                raise RuntimeError(f"server exited: {process.returncode}")
            try:
                if client.get("/health").status_code == 200:
                    return time.perf_counter() - started
            except httpx.HTTPError:
                pass
            time.sleep(2)
    raise TimeoutError("server did not become healthy")


def stop(process: subprocess.Popen, timeout: float = 120) -> None:
    """Stop the server's whole session and wait for the GPU to clear."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).split():
            return
        time.sleep(2)
    raise RuntimeError("GPU did not clear after stopping the server")
