"""Freeze, preflight, and compare the selected adaptive B200 precision recipes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)
from gleipnir.fouroversix_training import FOUROVERSIX_SDIST_SHA256, FOUROVERSIX_VERSION
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


def make_job(original: dict, output: Path, name: str, steps: int) -> dict:
    """Retain the frozen inputs and optimized non-precision recipe."""
    return {
        **original,
        "job_name": name,
        "design_role": "bounded_mlp_precision_diagnostic",
        "output_dir": str(output),
        "causal_adapter_dir": str(output / "causal_adapter"),
        "model_dir": str(output / "model"),
        "max_steps": steps,
        "micro_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "selective_torch_compile_canary_tokens": 0,
        "adaptive_microbatching": {
            "enabled": True,
            "max_padded_tokens": 16384,
            "max_micro_batch_size": 8,
            "profile": False,
        },
    }


def verify_inputs(job: dict) -> None:
    for path_key, hash_key in (
        ("student_rows", "student_rows_sha256"),
        ("soft_targets", "soft_targets_sha256"),
        ("selection_manifest", "selection_sha256"),
    ):
        if sha256_file(Path(job[path_key])) != job[hash_key]:
            raise ValueError(f"input checksum drift: {path_key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["steps"] != 10 or config["conditions"] not in [
        ["nf4", "bf16", "fouroversix"],
        ["nf4", "nf4-eager-mlp", "bf16", "fouroversix"],
    ]:
        raise ValueError("retain the bounded three-condition ten-update contract")
    if config["max_padded_tokens"] != 16384 or config["max_micro_batch_size"] != 8:
        raise ValueError("retain the selected adaptive recipe")
    original_jobs = [
        json.loads(line)
        for line in (ROOT / config["jobs"]).read_text().splitlines()
        if line
    ]
    original = next(
        j for j in original_jobs if j["job_name"] == config["condition_source"]
    )
    preflight_jobs = [
        json.loads(line)
        for line in (ROOT / config["preflight_jobs"]).read_text().splitlines()
        if line
    ]
    if len(preflight_jobs) != 1:
        raise ValueError("expected the frozen original longest-32 preflight")
    verify_inputs(original)
    verify_inputs(preflight_jobs[0])
    output = (ROOT / config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    logs = ROOT / config["logs"]
    logs.mkdir(parents=True, exist_ok=True)
    environment = triton_environment(
        DEFAULT_TRITON_TARGET,
        causal_conv1d_environment(
            DEFAULT_CAUSAL_CONV1D_TARGET, fla_environment(DEFAULT_FLA_TARGET)
        ),
    )
    kernel_target = ROOT / config["kernel_target"]
    environment["PYTHONPATH"] = f"{kernel_target}:{environment['PYTHONPATH']}"
    environment["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
    cache = ROOT / ".cache/training/qwen35_4b_b200_fa4/gpu-0"
    if not cache.is_dir():
        raise ValueError("the compatible persistent B200 compiler cache is missing")
    environment = gpu_environment(environment, 0, cache)
    environment["OMP_NUM_THREADS"] = "4"
    sources = [
        ROOT / "src/gleipnir/__init__.py",
        ROOT / "src/gleipnir/_compat.py",
        ROOT / "src/gleipnir/training/backends/fouroversix.py",
        ROOT / "src/gleipnir/training/screens/precision.py",
        ROOT / "experiments/deception_distillation/train_student_sft.py",
        *Path(__file__).parent.glob("*.py"),
        Path(__file__).with_name("README.md"),
    ]
    contract = {
        "config": config,
        "config_sha256": sha256_file(args.config),
        "source_revision": os.environ.get("GLEIPNIR_COMMIT"),
        "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sources},
        "original_jobs_sha256": sha256_file(ROOT / config["jobs"]),
        "preflight_jobs_sha256": sha256_file(ROOT / config["preflight_jobs"]),
        "original_job": original,
        "global_longest_job": preflight_jobs[0],
        "inputs_verified": True,
        "fouroversix": FOUROVERSIX_VERSION,
        "fouroversix_sdist_sha256": FOUROVERSIX_SDIST_SHA256,
        "compiler_cache": str(cache),
        "kernel_target": str(kernel_target),
        "authorization": (
            "user requested Four Over Six on existing B200 "
            "using selected optimized recipe"
        ),
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    status = {"status": "running", "stages": []}

    def publish():
        (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")

    def run_stage(name, command):
        stage = {
            "name": name,
            "command": command,
            "started_unix": time.time(),
            "status": "running",
        }
        status["stages"].append(stage)
        publish()
        print(f"starting_stage={name}", flush=True)
        with (logs / f"{name}.log").open("w") as handle:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        stage.update(
            status="complete" if result.returncode == 0 else "failed",
            returncode=result.returncode,
            ended_unix=time.time(),
        )
        publish()
        if result.returncode:
            raise RuntimeError(f"stage {name} failed; see {logs / f'{name}.log'}")

    def train(name, precision, source, steps):
        stage_output = output / name
        job = make_job(source, stage_output, name, steps)
        command = training_command(job)
        command.extend(
            [
                f"++student.quantization.mlp_precision={precision}",
                "++student.training.adapter_init_seed=0",
                "++student.training.precision_screen.enabled=true",
                f"++student.training.precision_screen.output_dir={stage_output}",
                f"++student.training.precision_screen.steps={steps}",
            ]
        )
        if name in config.get("eager_mlp_interfaces", []):
            command.append("++student.training.eager_mlp_interface=true")
        if steps == 1:
            command.append("student.training.warmup_ratio=0.0")
        job_file = output / f"{name}_job.json"
        job_file.write_text(
            json.dumps({"job": job, "command": command}, indent=2) + "\n"
        )
        run_stage(name, command)
        return json.loads((stage_output / "screen.json").read_text())

    try:
        run_stage(
            "kernel-canary",
            [
                sys.executable,
                str(Path(__file__).with_name("kernel_canary.py")),
                "--output",
                str(output / "kernel_canary.json"),
            ],
        )
        train("native-global-preflight", "fouroversix", preflight_jobs[0], 1)
        reports = {
            condition: train(
                condition,
                "nf4" if condition == "nf4-eager-mlp" else condition,
                original,
                10,
            )
            for condition in config["conditions"]
        }
        if len({r["initial_master_sha256"] for r in reports.values()}) != 1:
            raise ValueError(
                "precision conditions did not start with identical adapters"
            )
        if any(r["order"] != reports["nf4"]["order"] for r in reports.values()):
            raise ValueError("precision conditions changed update membership")
        summary = {
            "conditions": {
                precision: {
                    "training_seconds": r["training_seconds"],
                    "step_seconds": [step["seconds"] for step in r["steps"]],
                    "peak_allocated_gib": max(
                        step["peak_allocated_bytes"] for step in r["steps"]
                    )
                    / 2**30,
                    "before_common_loss": r["before_common_probe"]["mean_loss"],
                    "after_common_loss": r["after_common_probe"]["mean_loss"],
                    "native_calls": r["native_calls"],
                    "dynamo_graphs": r["dynamo_counters"]
                    .get("stats", {})
                    .get("unique_graphs"),
                }
                for precision, r in reports.items()
            },
            "native_speedup_over_nf4": reports["nf4"]["training_seconds"]
            / reports["fouroversix"]["training_seconds"],
            "native_speedup_over_bf16_mlp": reports["bf16"]["training_seconds"]
            / reports["fouroversix"]["training_seconds"],
            "quality_equivalence_established": False,
            "promotion": "none; short training probes and timing only",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        status["status"] = "complete"
        print(json.dumps(summary), flush=True)
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        publish()


if __name__ == "__main__":
    main()
