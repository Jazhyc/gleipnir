"""Serve a release target through its validated optimized B200 stack.

Replays the recorded server command (same flags, workers, receipts and merged
path) with the environment the recorded run used, then supervises the server
in the foreground. The intermittent CuTeDSL MXFP8 compile race
(``TYPE_UNSTABLE_JOIN``) is retried; any other startup failure is fatal.
Run with the serving interpreter, ``PYTHONPATH=src:.``, from the repository.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from experiments.release_serving.targets import (
    CONTEXT_LIMIT,
    RECORDED_CONTEXT,
    ROOT,
    SHARED_DIAGNOSTIC_SOURCES,
    TARGETS,
    Target,
)

STATE = Path("/tmp/gleipnir-release")
PRISTINE = Path("/opt/gleipnir-release/sources-27b")
COMPILE_RACE = "TYPE_UNSTABLE_JOIN"


def scheduler_binding(selection: dict) -> dict:
    from vllm.v1.core.sched import scheduler

    from gleipnir.data.monitoring import file_hash

    binding = {
        "upstream_sha256": file_hash(Path(inspect.getfile(scheduler))),
        "integration_sha256": file_hash(ROOT / "experiments/b200_vllm031/scheduler.py"),
    }
    if binding != selection["scheduler_binding"]:
        raise ValueError("selected pooling scheduler changed")
    return binding


def extend_context(command: list[str], env: dict, target: Target, limit: int) -> None:
    """Change only the context limit and bind the target's MXFP8 envelope receipt."""
    from gleipnir.data.monitoring import file_hash

    for flag in ("--max-model-len", "--max-num-batched-tokens"):
        if command[command.index(flag) + 1] != str(RECORDED_CONTEXT):
            raise ValueError("recorded context/chunk recipe changed")
    if limit == RECORDED_CONTEXT:
        return
    if not RECORDED_CONTEXT < limit <= CONTEXT_LIMIT:
        raise ValueError(
            f"context limit must be in ({RECORDED_CONTEXT}, {CONTEXT_LIMIT}]"
        )
    spec = target.long_context
    if file_hash(ROOT / spec.receipt) != spec.receipt_sha256:
        raise ValueError(f"long-context receipt changed: {spec.receipt}")
    command[command.index("--max-model-len") + 1] = str(limit)
    index = command.index("--additional-config") + 1
    additional = json.loads(command[index])
    if spec.worker is not None:
        worker = command.index("--worker-cls") + 1
        if command[worker] != spec.worker[0]:
            raise ValueError("recorded worker changed")
        command[worker] = spec.worker[1]
    for path in spec.sources:
        additional["gleipnir_frost_fp4"][path] = file_hash(ROOT / path)
    if spec.additional_key is not None:
        additional[spec.additional_key] = {
            "validation": spec.receipt,
            "validation_sha256": spec.receipt_sha256,
        }
    command[index] = json.dumps(additional, sort_keys=True)
    if spec.receipt_env is not None:
        env[spec.receipt_env] = spec.receipt


def prepare(
    target: Target, host: str, port: int, limit: int = CONTEXT_LIMIT
) -> tuple[list[str], dict]:
    from experiments.b200_vllm031.runtime import candidate_environment
    from experiments.b200_vllm031.source_bindings import restore_diagnostic_sources
    from gleipnir.serving.reference import selected_serving_default

    state = STATE / target.key
    state.mkdir(parents=True, exist_ok=True)
    command = json.loads((ROOT / target.server_receipt).read_text())["command"]
    command[0] = sys.executable
    command[command.index("--host") + 1] = host
    command[command.index("--port") + 1] = str(port)
    if command[command.index("--model") + 1] != str(target.merged_path):
        raise ValueError("recorded command and merged artifact disagree")
    if target.key == "27b":
        for relative in SHARED_DIAGNOSTIC_SOURCES:
            shutil.copy2(PRISTINE / relative, ROOT / relative)
    selection, _ = selected_serving_default(
        ROOT, last_query=target.launch == "optimized"
    )
    runtime = {n: importlib.metadata.version(n) for n in selection["runtime"]}
    if runtime != selection["runtime"]:
        raise ValueError("selected serving runtime changed")
    additional = json.loads(command[command.index("--additional-config") + 1])
    restore_diagnostic_sources(
        ROOT,
        additional["serving_condition"],
        ROOT / "results/b200_vllm031/pre_migration_sources.tar.gz",
    )
    parent = json.loads((ROOT / selection["host_parent"]).read_text())
    env = candidate_environment(ROOT)
    env.update(
        GLEIPNIR_GIGATOKEN_RECEIPT=str(state / "frontend.json"),
        VLLM_GDN_DECODE_KERNEL="cuda",
        GLEIPNIR_FLASHINFER_GDN_CP="auto",
        GLEIPNIR_VLLM031_SCHEDULER_BINDING=json.dumps(scheduler_binding(selection)),
    )
    if target.launch == "optimized":
        env["GLEIPNIR_FROST_WRAPPER_VALIDATION"] = parent["host_wrapper"]["validation"]
    else:
        env.pop("GLEIPNIR_FROST_WRAPPER_VALIDATION", None)
        env["GLEIPNIR_BF16_AUDIT"] = str(state / "bf16_audit.json")
    extend_context(command, env, target, limit)
    return command, env


def stream(process: subprocess.Popen, log_path: Path) -> None:
    """Copy server output to stdout and the attempt log."""
    with log_path.open("w") as log:
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        sys.stdout.flush()


def wait_ready(process: subprocess.Popen, port: int, timeout: float) -> bool:
    started = time.perf_counter()
    with httpx.Client(trust_env=False, timeout=10) as client:
        while time.perf_counter() - started < timeout:
            if process.poll() is not None:
                return False
            try:
                if client.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(2)
    raise TimeoutError("server startup exceeded the timeout")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(TARGETS))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=2400)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=CONTEXT_LIMIT,
        help=f"{RECORDED_CONTEXT} replays the recorded command unchanged",
    )
    args = parser.parse_args()
    target = TARGETS[args.model]
    command, env = prepare(target, args.host, args.port, args.max_model_len)
    for attempt in range(1, args.attempts + 1):
        log_path = STATE / target.key / f"server.{attempt}.log"
        print(
            f"gleipnir_serve_start {target.key} attempt={attempt} log={log_path}",
            flush=True,
        )
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        tee = threading.Thread(target=stream, args=(process, log_path), daemon=True)
        tee.start()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_, child=process: child.terminate())
        if wait_ready(process, args.port, args.startup_timeout):
            print(f"gleipnir_serve_ready {target.key} port={args.port}", flush=True)
            code = process.wait()
            tee.join()
            sys.exit(code)
        tee.join()
        tail = log_path.read_text(errors="replace")[-20000:]
        if COMPILE_RACE not in tail or attempt == args.attempts:
            raise SystemExit(f"server exited during startup ({process.returncode})")
        print(f"gleipnir_serve_retry compile race on attempt {attempt}", flush=True)


if __name__ == "__main__":
    main()
