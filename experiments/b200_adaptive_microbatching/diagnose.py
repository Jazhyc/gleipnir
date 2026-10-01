"""Run only the eight-input gradient canary, preserving the failed screen."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)
from gleipnir.monitoring_systems_screen import gpu_environment, sha256_file
from gleipnir.qwen35_fast_training import (
    DEFAULT_CAUSAL_CONV1D_TARGET,
    DEFAULT_FLA_TARGET,
    DEFAULT_TRITON_TARGET,
    causal_conv1d_environment,
    fla_environment,
    triton_environment,
)

ROOT = Path(__file__).resolve().parents[2]


def diagnostic_job(original: dict, output: Path, mode: str) -> dict:
    """Preserve data, adapter, loss, and seed; select eager or compiled canary."""
    return {
        **original,
        "job_name": f"diagnostic-{mode}",
        "design_role": "eight_input_gradient_canary_only",
        "output_dir": str(output),
        "causal_adapter_dir": str(output / "causal_adapter"),
        "model_dir": str(output / "model"),
        "max_steps": 1,
        "selective_torch_compile_policy": (
            "none" if mode == "eager" else original["selective_torch_compile_policy"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "compiled"), required=True)
    args = parser.parse_args()
    original_jobs = [
        json.loads(line) for line in args.jobs.read_text().splitlines() if line
    ]
    if len(original_jobs) != 1:
        raise ValueError("diagnostics require one frozen preflight job")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    job = diagnostic_job(original_jobs[0], output, args.mode)
    command = training_command(job)
    command.append("student.training.adaptive_microbatching.canary_only=true")
    contract = {
        "job": job,
        "command": command,
        "canary_only": True,
        "original_jobs_sha256": sha256_file(args.jobs),
        "revision": os.environ["GLEIPNIR_COMMIT"],
        "kernel_probe": "reuse preceding successful pinned-kernel preflight",
    }
    (output / "diagnostic_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    # Reuse the kernels verified by this screen without rebuilding or reprovisioning.
    environment = triton_environment(
        DEFAULT_TRITON_TARGET,
        causal_conv1d_environment(
            DEFAULT_CAUSAL_CONV1D_TARGET, fla_environment(DEFAULT_FLA_TARGET)
        ),
    )
    cache = ROOT / ".cache/training/qwen35_4b_b200_fa4/gpu-0"
    if not cache.is_dir():
        raise ValueError("diagnostics require the existing compatible B200 cache")
    subprocess.run(
        command, cwd=ROOT, env=gpu_environment(environment, 0, cache), check=True
    )


if __name__ == "__main__":
    main()
