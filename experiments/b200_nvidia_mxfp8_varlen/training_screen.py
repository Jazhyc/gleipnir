"""Bounded packed MXFP8 training against checksum-bound historical controls."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from experiments.b200_nvidia_mxfp8.run import environment
from experiments.b200_nvidia_mxfp8.training_screen import job_for as dense_job
from gleipnir.monitoring_systems_screen import sha256_file
from gleipnir.monitoring_training_command import training_command
from gleipnir.packed_benchmark import summarize

ROOT = Path(__file__).resolve().parents[2]


def accept_native(receipt: dict) -> dict:
    """Require actual native layout/execution correctness, retaining parity failure."""
    if receipt.get("status") != "execution_complete" or "exception" in receipt:
        raise ValueError("native execution did not complete")
    checks = receipt["quantizer_checks"]
    if len(checks) != 4 or {(r["heads"], r["columnwise"]) for r in checks} != {
        (16, False),
        (16, True),
        (4, False),
        (4, True),
    }:
        raise ValueError("incomplete packed quantizer checks")
    if not all(
        r[k] is True
        for r in checks
        for k in (
            "payload_bitexact",
            "canonical_scale_bitexact",
            "packed_scale_bitexact",
        )
    ):
        raise ValueError("packed quantization layout failed")
    comparison = receipt["native_dense_comparison"]
    if (
        not comparison["finite"]
        or comparison["forward_relative_l2"] != 0
        or (comparison["gradient_relative_l2"] != [0, 0, 0])
    ):
        raise ValueError("native packed/dense arithmetic differs")
    for gate in ("isolation", "graph_replay", "poisoned_dead_storage"):
        if receipt[gate]["passed"] is not True:
            raise ValueError(f"native {gate} failed")
    if receipt["poisoned_dead_storage"]["forward_relative_l2"] != 0 or (
        receipt["poisoned_dead_storage"]["gradient_relative_l2"] != [0, 0, 0]
    ):
        raise ValueError("poisoned buffers change native arithmetic")
    return {
        "execution_correct": True,
        "strict_parity_passed": receipt["fp32_comparison"]["strict_passed"],
        "fp32_comparison": receipt["fp32_comparison"],
    }


def job_for(config: dict, source: dict, recipe: dict) -> dict:
    """Keep the dense screen's optimizer/data contract and fresh startup gates."""
    job = dense_job(config, source, recipe, "nvidia_mxfp8")
    backend = config.get("candidate_backend", "nvidia_mxfp8_varlen")
    if backend not in {
        "nvidia_mxfp8_varlen",
        "nvidia_mxfp8_fused",
        "nvidia_mxfp8_square",
    }:
        raise ValueError("unsupported packed NVIDIA candidate")
    destination = ROOT / config["output"] / backend
    job.update(
        job_name=backend,
        packed_attention_backend=backend,
        output_dir=str(destination),
        causal_adapter_dir=str(destination / "causal_adapter"),
        model_dir=str(destination / "model"),
    )
    return job


def main(default_config: Path | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config
        or ROOT / "experiments/b200_nvidia_mxfp8_varlen/training_config.yaml",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    native_path = ROOT / config["native_reference"]
    if sha256_file(native_path) != config["native_reference_sha256"]:
        raise ValueError("native receipt checksum drift")
    backend = config.get("candidate_backend", "nvidia_mxfp8_varlen")
    native = json.loads(native_path.read_text())
    if backend in {"nvidia_mxfp8_fused", "nvidia_mxfp8_square"}:
        from experiments.b200_mxfp8_fused.training_screen import (
            accept_native as accept_fused,
        )

        acceptance = accept_fused(native, square=backend == "nvidia_mxfp8_square")
    else:
        acceptance = accept_native(native)
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
    controls = {}
    for name, contract in config["controls"].items():
        path = ROOT / contract["metadata"]
        if sha256_file(path) != contract["sha256"]:
            raise ValueError(f"historical control checksum drift: {name}")
        controls[name] = summarize(
            json.loads(path.read_text()),
            config["warmup_steps"],
            accept_learning=True,
            accept_timing=True,
        )
    source_paths = [
        args.config,
        Path(__file__),
        ROOT / "experiments/b200_nvidia_mxfp8_varlen/kernel_canary.py",
        ROOT / "experiments/b200_nvidia_mxfp8/run.py",
        ROOT / "experiments/b200_nvidia_mxfp8/training_screen.py",
        *ROOT.glob("src/gleipnir/nvidia_mxfp8_varlen*.py"),
        *ROOT.glob("src/gleipnir/nvidia_mxfp8_fused*.py"),
        *(ROOT / "experiments/b200_mxfp8_fused").glob("*.py"),
        *(
            ROOT / "src/gleipnir" / name
            for name in [
                "packed_sequences.py",
                "packed_training.py",
                "packed_training_screen.py",
                "packed_benchmark.py",
                "monitoring_training_command.py",
            ]
        ),
        ROOT / "experiments/deception_distillation/train_student_sft.py",
        ROOT / config["profile"],
    ]
    hashes = {}
    for path in source_paths:
        relative = path.relative_to(ROOT)
        destination = output / "executed_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        hashes[str(relative)] = sha256_file(path)
    recipe = yaml.safe_load((ROOT / config["profile"]).read_text())["recipe"]
    job = job_for(config, source, recipe)
    command = training_command(job) + [
        f"student.init_adapter={ROOT / config['initial_adapter']}",
        "++student.training.logging_steps=1",
    ]
    env = environment(config)
    report = {
        "status": "starting",
        "config": config,
        "native_acceptance": acceptance,
        "source_job": source,
        "job": job,
        "command": command,
        "source_sha256": hashes,
        "controls": controls,
        "cache_paths": {k: v for k, v in env.items() if "CACHE" in k},
    }

    def publish():
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    publish()
    started = time.perf_counter()
    try:
        with (logs / f"{backend}.log").open("x") as handle:
            subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        metadata_path = Path(job["causal_adapter_dir"]) / "training_metadata.json"
        metrics = summarize(
            json.loads(metadata_path.read_text()),
            config["warmup_steps"],
            accept_learning=True,
            accept_timing=bool(config.get("timing_authority")),
        )
        if metrics["initial_master_sha256"] != config["expected_initial_master_sha256"]:
            raise ValueError("initial adapter master checksum drift")
        for name, control in controls.items():
            if metrics["physical_contract"] != control["physical_contract"]:
                raise ValueError(f"physical rows differ from historical {name}")
        report.update(
            status="complete",
            wall_seconds=time.perf_counter() - started,
            candidate=metrics,
            metadata_sha256=sha256_file(metadata_path),
            measured_time_reduction={
                name: 1
                - metrics["measured_total_seconds"] / c["measured_total_seconds"]
                for name, c in controls.items()
            },
            default_promoted=False,
            quality_validated=False,
        )
        publish()
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        publish()
        raise


if __name__ == "__main__":
    main()
