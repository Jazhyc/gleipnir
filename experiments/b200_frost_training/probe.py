"""Deferred GPU gate and paired complete-MLP timings for direct training bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).parent
GEOMETRIES = {(2560, 18432), (18432, 2560), (9216, 2560), (2560, 9216)}


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def accept_checks(checks: dict) -> None:
    """CPU-side gate for exact outputs, seven gradients and replay/stream checks."""
    flags = (
        "output_exact",
        "gradients_exact",
        "finite",
        "independent_outputs",
        "changed_input_replay_exact",
        "changed_input_replay_changed",
        "changed_master_replay_exact",
        "changed_master_replay_changed",
        "alternate_stream_exact",
    )
    if (
        any(checks.get(key) is not True for key in flags)
        or checks.get("gradient_count") != 7
    ):
        raise ValueError("complete-MLP binding parity failed")
    geometries = {
        (p["k"], p["n"]) for p in checks["dispatch"]["geometries"] if p["calls"] > 0
    }
    if geometries != GEOMETRIES:
        raise ValueError("forward/input-gradient binding coverage incomplete")


def require_exclusive_gpu(processes: str, worker_pid: int) -> None:
    """Reject concurrent users; this probe never stops an existing GPU process."""
    pids = {int(line.strip()) for line in processes.splitlines() if line.strip()}
    if pids - {worker_pid}:
        raise RuntimeError("GPU has another process; run the probe on an idle B200")


def worker(output: Path, settings: dict) -> None:
    report: dict[str, Any] = dict(
        status="starting",
        shapes=[],
        synthetic_weights=True,
        full_backward_included=True,
        full_training=False,
        automatic_promotion=False,
        arithmetic_changed=False,
        settings=settings,
    )
    receipt = output / "probe.json"
    write(receipt, report)
    try:
        import torch
        from transformers import AutoConfig

        from experiments.b200_mlp_gemm.probe import make_mlp
        from gleipnir.kernels.fp4.cudnn_fp4_mlp import install_fp4_mlp
        from gleipnir.kernels.fp4.frost_bindings import training_frost_bindings
        from gleipnir.training.bf16_lora import configure_bf16_reductions

        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            10,
            0,
        ):
            raise ValueError("the native gate requires the authorized B200")
        processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        )
        require_exclusive_gpu(processes, os.getpid())
        torch.manual_seed(settings["seed"])
        configure_bf16_reductions(allow_reduced_precision=False, allow_split_k=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        from torch._inductor import config as inductor_config

        inductor_config.emulate_precision_casts = True
        config = AutoConfig.from_pretrained(
            settings["model"], revision=settings["revision"], local_files_only=True
        ).text_config
        if (config.hidden_size, config.intermediate_size) != (2560, 9216):
            raise ValueError("training geometry drift")
        module = make_mlp(config)
        parameters = tuple(p for p in module.parameters() if p.requires_grad)
        if len(parameters) != 6 or any(p.dtype != torch.float32 for p in parameters):
            raise ValueError("expected six live FP32 adapter masters")
        report["installation"] = install_fp4_mlp(
            module, hardware_packing=True, fused_descale=True
        )
        # Explicit graphs test replay; ordinary samples execute host dispatch.
        forward = torch.compile(
            module, fullgraph=True, dynamic=True, options={"triton.cudagraphs": False}
        )
        report["runtime"] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compile_cudagraphs": False,
        }

        def snapshot(result):
            y, grads = result
            return (y.detach().clone(), tuple(g.detach().clone() for g in grads))

        def exact(a, b):
            return torch.equal(a[0], b[0]) and all(
                torch.equal(x, y) for x, y in zip(a[1], b[1], strict=True)
            )

        with training_frost_bindings("original") as control:
            for rows in settings["rows"]:
                x = torch.randn(
                    1,
                    rows,
                    2560,
                    device="cuda",
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                dy = torch.randn_like(x) / (2560**0.5)

                def step(x=x, dy=dy):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        y = forward(x)
                    return y, torch.autograd.grad(y, (x, *parameters), dy)

                for _ in range(settings["warmups"]):
                    for mode in ("original", "direct"):
                        control.set_mode(mode)
                        step()
                torch.cuda.synchronize()
                control.set_mode("original")
                reference = snapshot(step())
                control.set_mode("direct")
                direct = step()
                candidate = snapshot(direct)
                saved_output = direct[0].detach().clone()
                second = step()
                independent = direct[0].data_ptr() != second[
                    0
                ].data_ptr() and torch.equal(direct[0], saved_output)
                flags = dict(
                    output_exact=torch.equal(reference[0], candidate[0]),
                    gradients_exact=all(
                        torch.equal(a, b)
                        for a, b in zip(reference[1], candidate[1], strict=True)
                    ),
                    gradient_count=len(candidate[1]),
                    finite=all(
                        bool(torch.isfinite(t).all())
                        for t in (candidate[0], *candidate[1])
                    ),
                    independent_outputs=independent,
                )
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    stream_result = snapshot(step())
                torch.cuda.current_stream().wait_stream(side)
                flags["alternate_stream_exact"] = exact(candidate, stream_result)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = step()
                original_x = x.detach().clone()
                with torch.no_grad():
                    x.mul_(0.9)
                graph.replay()
                replay = snapshot(captured)
                control.set_mode("original")
                changed_reference = snapshot(step())
                flags["changed_input_replay_exact"] = exact(replay, changed_reference)
                flags["changed_input_replay_changed"] = not torch.equal(
                    candidate[0], changed_reference[0]
                )
                master = parameters[1].detach().clone()
                with torch.no_grad():
                    parameters[1].add_(0.01)
                graph.replay()
                replay_master = snapshot(captured)
                master_reference = snapshot(step())
                flags["changed_master_replay_exact"] = exact(
                    replay_master, master_reference
                )
                flags["changed_master_replay_changed"] = not torch.equal(
                    changed_reference[0], master_reference[0]
                )
                with torch.no_grad():
                    x.copy_(original_x)
                    parameters[1].copy_(master)
                flags["dispatch"] = control.state()
                row = dict(
                    rows=rows,
                    checks=flags,
                    timing_scope=(
                        "ordinary synchronized complete MLP forward "
                        "and all seven gradients"
                    ),
                )
                report["shapes"].append(row)
                write(receipt, report)
                accept_checks(flags)
                # Replay gates precede timing; modes switch outside timing.
                for _ in range(settings["warmups"]):
                    for mode in ("original", "direct"):
                        control.set_mode(mode)
                        step()
                samples = {"original": [], "direct": []}
                for repeat in range(settings["repeats"]):
                    order = (
                        ("original", "direct")
                        if repeat % 2 == 0
                        else ("direct", "original")
                    )
                    for mode in order:
                        control.set_mode(mode)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        step()
                        torch.cuda.synchronize()
                        samples[mode].append(1000 * (time.perf_counter() - start))
                row["samples_ms"] = samples
                row["mean_ms"] = {
                    mode: statistics.mean(v) for mode, v in samples.items()
                }
                row["step_time_change_percent"] = 100 * (
                    row["mean_ms"]["direct"] / row["mean_ms"]["original"] - 1
                )
                write(receipt, report)
                print(
                    "binding_probe_shape",
                    rows,
                    row["step_time_change_percent"],
                    flush=True,
                )
                del captured, graph, direct, candidate, reference, second, stream_result
        report.update(status="complete", passed=True, restored_bindings=control.state())
        write(receipt, report)
    except BaseException as error:
        report.update(
            status="failed",
            passed=False,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        write(receipt, report)
        raise


def stop_worker(process: subprocess.Popen) -> None:
    """Stop only this probe's new process group, tolerating an exit race."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.name.isalnum():
        raise ValueError("probe name must be alphanumeric")
    settings = json.loads((EXPERIMENT / "config.json").read_text())
    output = ROOT / "results/b200_frost_training" / args.name
    if args.worker:
        worker(output, settings)
        return
    output.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        EXPERIMENT / "config.json",
        ROOT / "src/gleipnir/kernels/fp4/frost_bindings.py",
        ROOT / "src/gleipnir/kernels/fp4/cudnn_fp4_epilogue.py",
        ROOT / "src/gleipnir/kernels/fp4/cudnn_fp4_mlp.py",
        ROOT / "experiments/b200_mlp_gemm/probe.py",
    ]
    bindings = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = output / "executed_sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        bindings[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    write(output / "source_binding.json", bindings)
    with (output / "probe.log").open("x") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experiments.b200_frost_training.probe",
                "--name",
                args.name,
                "--worker",
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=settings["max_seconds"])
        except BaseException as error:
            stop_worker(process)
            write(
                output / "failure.json", {"error": f"{type(error).__name__}: {error}"}
            )
            raise
    if code:
        write(output / "failure.json", {"worker_exit_code": code, "log": "probe.log"})
        raise SystemExit(code)
    print("binding_probe_complete", output, flush=True)


if __name__ == "__main__":
    main()
