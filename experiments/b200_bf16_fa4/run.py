"""Run matched ordinary packed Trainer trajectories with durable caches."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

from gleipnir.flashqla_training import flashqla_environment
from gleipnir.monitoring_systems_screen import gpu_environment, sha256_file
from gleipnir.monitoring_training_command import training_command
from gleipnir.qwen35_fast_training import (
    DEFAULT_CAUSAL_CONV1D_TARGET,
    DEFAULT_FLA_TARGET,
    DEFAULT_TRITON_TARGET,
    causal_conv1d_environment,
    fla_environment,
    triton_environment,
)

ROOT = Path(__file__).resolve().parents[2]


def benchmark_job(config: dict, source: dict, recipe: dict, backend: str) -> dict:
    """Keep the frozen data and recipe, selecting only the packed attention route."""
    if config["steps"] != 20 or config["warmup_steps"] != 10:
        raise ValueError("benchmark requires ten warmup and ten measured updates")
    if backend not in {"sdpa", "flash_attention_4"}:
        raise ValueError("unsupported benchmark backend")
    destination = ROOT / config["output"] / backend
    job = {
        **source,
        **recipe,
        "job_name": backend,
        "max_steps": config["steps"],
        "save_steps": 1000000,
        "output_dir": str(destination),
        "causal_adapter_dir": str(destination / "causal_adapter"),
        "model_dir": str(destination / "model"),
        "hydra_log_dir": str(ROOT / config["logs"]),
        "packed_attention_backend": backend,
    }
    job.pop("startup_validation_reference", None)
    if backend == "sdpa":
        job["startup_validation_reference"] = str(ROOT / config["validation_reference"])
    else:
        job["packed_attention_version"] = config["fa4_version"]
    return job


def summarize(metadata: dict, warmup: int) -> dict:
    """Count every measured update and preserve its physical execution contract."""
    durations = metadata["optimizer_step_timing"]["durations_seconds"]
    if len(durations) != 20 or metadata["training_state"]["global_step"] != 20:
        raise ValueError("incomplete benchmark trajectory")
    if any(not math.isfinite(t) or t <= 0 for t in durations):
        raise ValueError("nonfinite/nonpositive benchmark duration")
    packing = metadata["sequence_packing"]
    if not packing.get("startup_validation") and not all(
        packing[key]["passed"]
        for key in ["eager_canary", "compiled_canary", "preflight"]
    ):
        raise ValueError("fresh packing gates did not pass")
    if (
        packing["initial_master_sha256"] == packing["final_master_sha256"]
        or metadata["quantization"]["enabled"]
        or metadata["checkpointed_layer_indices"]
    ):
        raise ValueError("benchmark precision/update/checkpoint contract drift")
    records = metadata["adaptive_microbatching"]["records"]
    if any(r["tokens"] != r["padded_tokens"] for r in records):
        raise ValueError("benchmark introduced padding")
    steady = durations[warmup:]
    return {
        "durations_seconds": durations,
        "measured_total_seconds": sum(steady),
        "measured_mean_seconds": statistics.mean(steady),
        "all_update_seconds": sum(durations),
        "trainer_loop_seconds": metadata["train_metrics"]["train_runtime"],
        "peak_allocated_gib": metadata["peak_cuda_memory_allocated_bytes"] / 2**30,
        "physical_contract": [
            {
                k: row[k]
                for k in ["update", "logical_indices", "tokens", "padded_tokens"]
            }
            for row in records
        ],
        "attention": {
            k: packing.get(k)
            for k in [
                "full_attention",
                "attention_backend",
                "attention_version",
            ]
        },
        "startup_validation": metadata.get("startup_validation"),
        "initial_master_sha256": packing["initial_master_sha256"],
        "compiled": metadata["selective_torch_compile"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/b200_bf16_fa4/config.yaml")
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    logs = ROOT / config["logs"]
    logs.mkdir(parents=True, exist_ok=True)
    sources = [
        json.loads(line)
        for line in (ROOT / config["source_jobs"]).read_text().splitlines()
        if line
    ]
    (source,) = [s for s in sources if s["job_name"] == config["condition_source"]]
    for path_key, hash_key in [
        ("student_rows", "student_rows_sha256"),
        ("soft_targets", "soft_targets_sha256"),
        ("selection_manifest", "selection_sha256"),
    ]:
        if sha256_file(Path(source[path_key])) != source[hash_key]:
            raise ValueError(f"input checksum drift: {path_key}")
    profile = yaml.safe_load((ROOT / config["profile"]).read_text())
    initial = ROOT / config["initial_adapter"]
    environment = flashqla_environment(
        triton_environment(
            DEFAULT_TRITON_TARGET,
            causal_conv1d_environment(
                DEFAULT_CAUSAL_CONV1D_TARGET,
                fla_environment(DEFAULT_FLA_TARGET, dict(os.environ)),
            ),
        )
    )
    # Retain FlashQLA's pinned TVM FFI; FA4's CuTe modules are isolated by
    # sitecustomize. Validate their combined runtime before any model load.
    environment["PYTHONPATH"] += ":" + str(ROOT / config["fa4_overlay"])
    environment = gpu_environment(environment, 0, ROOT / config["compiler_cache"])
    environment["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"
    environment["FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"] = str(ROOT / config["fa4_cache"])
    environment["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
    environment["OMP_NUM_THREADS"] = "4"
    files = [
        args.config,
        Path(__file__),
        ROOT / config["profile"],
        ROOT / "src/gleipnir/packed_sequences.py",
        ROOT / "src/gleipnir/packed_training.py",
        ROOT / "experiments/deception_distillation/train_student_sft.py",
    ]
    report = {
        "status": "running",
        "config": config,
        "source_job": source,
        "source_sha256": {str(p): sha256_file(p) for p in files},
        "initial_adapter_files": {
            p.name: sha256_file(p) for p in initial.iterdir() if p.is_file()
        },
        "cache_paths": {k: v for k, v in environment.items() if "CACHE" in k},
        "conditions": {},
    }

    def publish():
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    publish()
    try:
        # Same dependency overlay for both conditions. No reinstall or cold cache.
        with (logs / "native-canary.log").open("w") as handle:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.b200_bf16_fa4.kernel_canary",
                    "--output",
                    str(output / "kernel_canary.json"),
                ],
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        for backend in config["conditions"]:
            job = benchmark_job(config, source, profile["recipe"], backend)
            command = training_command(job) + [
                f"student.init_adapter={initial}",
                "++student.training.logging_steps=1",
            ]
            report["conditions"][backend] = {
                "status": "running",
                "job": job,
                "command": command,
            }
            publish()
            print(f"condition_start={backend}", flush=True)
            started = time.perf_counter()
            with (logs / f"{backend}.log").open("w") as handle:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            metadata = json.loads(
                (Path(job["causal_adapter_dir"]) / "training_metadata.json").read_text()
            )
            summary = summarize(metadata, config["warmup_steps"])
            if (
                summary["initial_master_sha256"]
                != config["expected_initial_master_sha256"]
            ):
                raise ValueError("initial master identity drift")
            report["conditions"][backend].update(
                status="complete",
                wall_seconds=time.perf_counter() - started,
                **summary,
            )
            publish()
            print(
                f"condition_complete={backend} "
                f"mean_seconds={summary['measured_mean_seconds']}",
                flush=True,
            )
        control, candidate = [
            report["conditions"][k] for k in ["sdpa", "flash_attention_4"]
        ]
        if candidate["physical_contract"] != control["physical_contract"]:
            raise ValueError("physical batches/tokens do not match")
        gain = (
            1 - candidate["measured_total_seconds"] / control["measured_total_seconds"]
        )
        loop_gain = (
            1 - candidate["trainer_loop_seconds"] / control["trainer_loop_seconds"]
        )
        report.update(
            status="complete",
            measured_time_reduction=gain,
            complete_update_time_reduction=loop_gain,
            promising=min(gain, loop_gain) >= config["minimum_relative_improvement"],
            adoption="requires_reverse_order_replication_and_quality_validation",
        )
        publish()
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        publish()
        raise


if __name__ == "__main__":
    main()
