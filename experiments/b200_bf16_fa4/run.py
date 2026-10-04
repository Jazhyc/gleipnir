"""Run matched ordinary packed Trainer trajectories with durable caches."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from gleipnir.monitoring_systems_screen import sha256_file
from gleipnir.monitoring_training_command import training_command
from gleipnir.packed_benchmark import benchmark_environment as packed_environment
from gleipnir.packed_benchmark import summarize as summarize_trajectory

ROOT = Path(__file__).resolve().parents[2]


def benchmark_environment(config: dict) -> dict[str, str]:
    """Use the same pinned overlays and existing persistent caches for every stage."""
    return packed_environment(
        {**config, "kernel_overlays": [config["fa4_overlay"]]}, ROOT
    )


def run_native_canary(logs: Path, output: Path, environment: dict[str, str]) -> None:
    """Record a fresh native receipt when no unchanged validation is available."""
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
        if tolerance := config.get("candidate_learning_gradient_tolerance"):
            job["packing_learning_gradient_tolerance"] = tolerance
    return job


def summarize(metadata: dict, warmup: int) -> dict:
    """Preserve explicit FA4 learning acceptance separately from strict gates."""
    return summarize_trajectory(metadata, warmup, accept_learning=True)


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
    environment = benchmark_environment(config)
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
        native_reference = config.get("native_canary_reference")
        if native_reference:
            reference = ROOT / native_reference
            native = json.loads(reference.read_text())
            if (
                not native["passed"]
                or native["packages"]["flash-attn-4"] != config["fa4_version"]
            ):
                raise ValueError("native canary reference is incompatible")
            report["native_canary"] = {
                "reused": True,
                "performed": False,
                "reference": str(reference),
                "sha256": sha256_file(reference),
            }
            publish()
        else:
            run_native_canary(logs, output, environment)
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
