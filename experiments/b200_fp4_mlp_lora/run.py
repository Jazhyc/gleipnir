"""Screen native FP4 MLP LoRA against a frozen completed BF16 control."""

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
from gleipnir.packed_benchmark import benchmark_environment, summarize

ROOT = Path(__file__).resolve().parents[2]


def candidate_job(config: dict, baseline: dict) -> dict:
    """Change MLP arithmetic while preserving the complete frozen training job."""
    job = dict(baseline["job"])
    for key in (
        "startup_validation_reference",
        "packed_attention_version",
        "packing_learning_gradient_tolerance",
    ):
        job.pop(key, None)
    output = ROOT / config["output"] / "fp4_mlp"
    job.update(
        job_name="fp4_mlp",
        full_bf16_lora=False,
        fp4_mlp_lora=True,
        fp4_backward_mode="dequantized_bf16",
        fp4_row_scaled_activations=True,
        fp4_fused_row_scaling=True,
        packed_attention_backend="sdpa",
        expected_initial_master_sha256=config["expected_initial_master_sha256"],
        output_dir=str(output),
        causal_adapter_dir=str(output / "causal_adapter"),
        model_dir=str(output / "model"),
        hydra_log_dir=str(ROOT / config["logs"]),
        max_steps=config["steps"],
        save_steps=1000000,
    )
    if config["steps"] != 20 or config["warmup_steps"] != 10:
        raise ValueError("screen requires ten warmup and ten measured updates")
    return job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/b200_fp4_mlp_lora/config.yaml")
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    source = ROOT / config["baseline"]
    original = json.loads(source.read_text())
    control = original["conditions"][config["baseline_condition"]]
    if original["status"] != "complete" or control["status"] != "complete":
        raise ValueError("baseline is incomplete")
    if control["initial_master_sha256"] != config["expected_initial_master_sha256"]:
        raise ValueError("baseline initial master mismatch")
    initial = ROOT / config["initial_adapter"]
    for name, digest in original["initial_adapter_files"].items():
        if sha256_file(initial / name) != digest:
            raise ValueError(f"initial adapter file drift: {name}")
    job = candidate_job(config, control)
    for path_key, hash_key in [
        ("student_rows", "student_rows_sha256"),
        ("soft_targets", "soft_targets_sha256"),
        ("selection_manifest", "selection_sha256"),
    ]:
        if sha256_file(Path(job[path_key])) != job[hash_key]:
            raise ValueError(f"input identity drift: {path_key}")
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    logs = ROOT / config["logs"]
    logs.mkdir(parents=True, exist_ok=True)
    environment = benchmark_environment(config, ROOT)
    command = training_command(job) + [
        f"student.init_adapter={initial}",
        "++student.training.logging_steps=1",
    ]
    source_files = [
        args.config,
        Path(__file__),
        ROOT / "experiments/deception_distillation/train_student_sft.py",
        ROOT / "src/gleipnir/bf16_lora.py",
        ROOT / "src/gleipnir/fouroversix_training.py",
        ROOT / "src/gleipnir/fp4_row_kernels.py",
        ROOT / "src/gleipnir/training/packed.py",
        ROOT / "src/gleipnir/__init__.py",
        ROOT / "src/gleipnir/_compat.py",
        ROOT / "src/gleipnir/packed_benchmark.py",
        ROOT / "src/gleipnir/campaigns/training_command.py",
        ROOT / "src/gleipnir/packed_sequences.py",
        ROOT / "src/gleipnir/packed_training_screen.py",
        ROOT / "experiments/b200_fouroversix/kernel_canary.py",
        ROOT / "experiments/fp4_stability/row_kernel_canary.py",
    ]
    report = {
        "status": "running",
        "config": config,
        "job": job,
        "command": command,
        "baseline_reference": str(source),
        "baseline_sha256": sha256_file(source),
        "baseline": control,
        "source_sha256": {str(p): sha256_file(p) for p in source_files},
        "cache_paths": {k: v for k, v in environment.items() if "CACHE" in k},
        "initial_adapter_files": original["initial_adapter_files"],
        "stages": {},
    }
    for path in source_files:
        absolute = path if path.is_absolute() else ROOT / path
        archived = output / "executed_sources" / absolute.relative_to(ROOT)
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(absolute.read_bytes())

    def publish() -> None:
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    def stage(name: str, invocation: list[str]) -> float:
        report["stages"][name] = {"status": "running", "command": invocation}
        publish()
        print(f"stage_start={name}", flush=True)
        started = time.perf_counter()
        with (logs / f"{name}.log").open("w") as handle:
            subprocess.run(
                invocation,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        elapsed = time.perf_counter() - started
        report["stages"][name].update(status="complete", wall_seconds=elapsed)
        publish()
        return elapsed

    publish()
    try:
        stage(
            "row_canary",
            [
                sys.executable,
                "-m",
                "experiments.fp4_stability.row_kernel_canary",
                "--output",
                str(output / "row_canary.json"),
            ],
        )
        stage(
            "native_canary",
            [
                sys.executable,
                "-m",
                "experiments.b200_fouroversix.kernel_canary",
                "--output",
                str(output / "native_canary.json"),
                "--backward-mode",
                "dequantized_bf16",
                "--row-scaled-activations",
                "--fused-row-scaling",
            ],
        )
        elapsed = stage("training", command)
        metadata = json.loads(
            (Path(job["causal_adapter_dir"]) / "training_metadata.json").read_text()
        )
        candidate = summarize(metadata, config["warmup_steps"])
        calls = metadata["native_mlp_calls"]
        if not (
            calls["modules"] == 96
            and calls["forward"] > 0
            and calls["dequantized_bf16_backward"] > 0
            and calls["fp4_backward"] == 0
            and metadata["quantization"]["fp4_mlp_lora"]["verified"]
        ):
            raise ValueError("native FP4 execution contract failed")
        if candidate["physical_contract"] != control["physical_contract"]:
            raise ValueError("physical batches/token counts differ from baseline")
        if (
            candidate["initial_master_sha256"]
            != config["expected_initial_master_sha256"]
        ):
            raise ValueError("initial master mismatch")
        gain = (
            1 - candidate["measured_total_seconds"] / control["measured_total_seconds"]
        )
        report.update(
            status="complete",
            candidate=candidate,
            native_calls=calls,
            wall_seconds=elapsed,
            measured_time_reduction=gain,
            promising=gain >= config["minimum_relative_improvement"],
            adoption="requires_replication_and_quality_validation",
        )
        publish()
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        for value in report["stages"].values():
            if value["status"] == "running":
                value.update(status="failed", error=report["error"])
        receipt = Path(job["causal_adapter_dir"]) / "packing_canary.json"
        if receipt.exists():
            report["failed_packing_receipt"] = json.loads(receipt.read_text())
        publish()
        raise


if __name__ == "__main__":
    main()
