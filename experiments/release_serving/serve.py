"""Serve a release target through its validated optimized B200 stack.

Replays the recorded server command (same flags, workers, receipts and merged
path) with the environment the recorded run used, then supervises the server
in the foreground. The intermittent CuTeDSL MXFP8 compile race
(``TYPE_UNSTABLE_JOIN``) is retried; any other startup failure is fatal.
On a B300 the same stack runs through ``hardware_hook`` after the attention
kernel is compiled outside the server (``compile_attention``).
Run with the serving interpreter, ``PYTHONPATH=src:.``, from the repository.
"""

from __future__ import annotations

import argparse
import importlib.metadata
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
    PREFIX_CACHE_FLAGS,
    PREFIX_WORKER_SOURCE,
    RECORDED_CONTEXT,
    ROOT,
    SHARED_DIAGNOSTIC_SOURCES,
    TARGETS,
    Target,
)

STATE = Path("/tmp/gleipnir-release")
GRAPHS = "--compilation-config"
TOKENIZER_HOOK = Path(__file__).resolve().parent / "tokenizer_hook"
PRISTINE = Path("/opt/gleipnir-release/sources-27b")
COMPILE_RACE = "TYPE_UNSTABLE_JOIN"
HARDWARE_HOOK = Path(__file__).resolve().parent / "hardware_hook"
# The GPU of the kernel receipts, and its compute capability.
VALIDATED_GPU = ("NVIDIA B200", "10.0")
# Other GPUs that run the validated kernels with identical scores.
COMPATIBLE_GPUS = ("NVIDIA B300 SXM6 AC",)
# Query and key/value heads of the targets whose attention plan is derived
# from the model config.
ATTENTION_HEADS = {"27b": (24, 4)}


def scheduler_binding(selection: dict) -> dict:


    from gleipnir.serving.sources import scheduler_binding as live_binding

    return live_binding(ROOT, selection["scheduler_binding"])


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


def enable_prefix_caching(command: list[str], target: Target) -> None:
    """Switch to align-mode prefix caching; scores then drift (opt-in only)."""
    from gleipnir.data.monitoring import file_hash

    if "--no-enable-prefix-caching" not in command:
        raise ValueError("recorded command no longer disables prefix caching")
    index = command.index("--no-enable-prefix-caching")
    command[index : index + 1] = PREFIX_CACHE_FLAGS
    if target.prefix_workers:
        worker = command.index("--worker-cls") + 1
        if command[worker] not in target.prefix_workers:
            raise ValueError(f"no prefix-cached twin for {command[worker]}")
        command[worker] = target.prefix_workers[command[worker]]
        index = command.index("--additional-config") + 1
        additional = json.loads(command[index])
        additional["gleipnir_frost_fp4"][PREFIX_WORKER_SOURCE] = file_hash(
            ROOT / PREFIX_WORKER_SOURCE
        )
        command[index] = json.dumps(additional, sort_keys=True)


def add_graphs(command: list[str], target: Target) -> None:
    """Capture the target's larger CUDA graphs (scores stay bit-identical)."""
    if not target.graph_sizes:
        return
    if GRAPHS in command:
        raise ValueError("recorded command already sets a compilation config")
    sizes = {"cudagraph_capture_sizes": list(target.graph_sizes)}
    command += [GRAPHS, json.dumps(sizes)]


def enable_native_tokenizer(env: dict, state: Path) -> None:
    """Make the installed Gigatoken encoder effective on vLLM 0.31."""
    env.update(
        GLEIPNIR_TOKENIZER_HOOK="1",
        GLEIPNIR_TOKENIZER_RECEIPT=str(state / "tokenizer.json"),
        PYTHONPATH=f"{TOKENIZER_HOOK}:{env['PYTHONPATH']}",
    )


def gpu_name() -> str:
    """Model name of the first GPU, without creating a CUDA context."""
    query = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    listing = subprocess.run(query, check=True, capture_output=True, text=True)
    return listing.stdout.splitlines()[0].strip()


def enable_compatible_gpu(env: dict, gpu: str) -> bool:
    """Route a compatible GPU through ``hardware_hook``; True if it applies."""
    if gpu not in COMPATIBLE_GPUS:
        return False
    env.update(
        GLEIPNIR_GPU_ALIAS=VALIDATED_GPU[0],
        GLEIPNIR_GPU_CAPABILITY=VALIDATED_GPU[1],
        PYTHONPATH=f"{HARDWARE_HOOK}:{env['PYTHONPATH']}",
    )
    return True


def attention_compile_command(target: Target, limit: int) -> list[str]:
    module = "experiments.release_serving.compile_attention"
    command = [sys.executable, "-m", module, "--length", str(limit)]
    if target.key in ATTENTION_HEADS:
        command += ["--heads", *map(str, ATTENTION_HEADS[target.key])]
    return command


def compile_attention(target: Target, env: dict, limit: int) -> None:
    """Build the attention kernel for this GPU before the server needs it."""
    log_path = STATE / target.key / "compile_attention.log"
    started = time.perf_counter()
    with log_path.open("w") as log:
        code = subprocess.call(
            attention_compile_command(target, limit),
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if code:
        raise SystemExit(f"attention kernel compile failed ({code}); see {log_path}")
    seconds = time.perf_counter() - started
    print(f"gleipnir_serve_attention_compiled seconds={seconds:.0f}", flush=True)


def prepare(
    target: Target,
    host: str,
    port: int,
    limit: int = CONTEXT_LIMIT,
    prefix_caching: bool = False,
    graphs: bool = False,
    native_tokenizer: bool = False,
) -> tuple[list[str], dict]:
    """The recorded command and environment, plus the image's serving options.

    ``graphs`` and ``native_tokenizer`` are what the image adds since 1.0;
    without them (the default) the recorded command is replayed, which the
    serving experiments rely on.
    """
    from gleipnir.serving.reference import selected_serving_default
    from gleipnir.serving.sources import promote_command
    from gleipnir.serving.stack.environment import candidate_environment
    from gleipnir.serving.stack.source_bindings import restore_diagnostic_sources

    state = STATE / target.key
    state.mkdir(parents=True, exist_ok=True)
    command = json.loads((ROOT / target.server_receipt).read_text())["command"]
    command[0] = sys.executable
    command[command.index("--host") + 1] = host
    command[command.index("--port") + 1] = str(port)
    if command[command.index("--model") + 1] != str(target.merged_path):
        raise ValueError("recorded command and merged artifact disagree")
    # The recorded command names the experiment modules it was validated with;
    # serve it on the promoted gleipnir.serving.stack after verifying sources.
    command = promote_command(ROOT, command)
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
    if prefix_caching:
        enable_prefix_caching(command, target)
    if graphs:
        add_graphs(command, target)
    if native_tokenizer:
        enable_native_tokenizer(env, state)
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
    parser.add_argument(
        "--prefix-caching",
        action="store_true",
        help="opt-in align-mode prefix caching for growing sessions; cached "
        "scores drift from uncached by up to about 0.15, so keep it off for "
        "evaluation",
    )
    parser.add_argument(
        "--recorded-graphs",
        action="store_true",
        help="keep the CUDA graph sizes of the recorded command (4B: up to 256)",
    )
    parser.add_argument(
        "--transformers-tokenizer",
        action="store_true",
        help="tokenize with Transformers, as stock vLLM does, not Gigatoken",
    )
    args = parser.parse_args()
    target = TARGETS[args.model]
    command, env = prepare(
        target,
        args.host,
        args.port,
        args.max_model_len,
        args.prefix_caching,
        graphs=not args.recorded_graphs,
        native_tokenizer=not args.transformers_tokenizer,
    )
    gpu = gpu_name()
    if enable_compatible_gpu(env, gpu):
        print(f"gleipnir_serve_gpu {gpu} served as {VALIDATED_GPU[0]}", flush=True)
        compile_attention(target, env, args.max_model_len)
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
