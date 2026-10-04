"""Compare complete MXFP8/FA4 Trainer updates, preserving strict native failures."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from experiments.b200_nvidia_mxfp8.run import environment
from gleipnir.monitoring_systems_screen import sha256_file
from gleipnir.monitoring_training_command import training_command
from gleipnir.packed_benchmark import summarize

ROOT = Path(__file__).resolve().parents[2]


def accept_native(receipt: dict, config: dict) -> dict:
    """Accept only bounded quantization differences; never relabel strict gates."""
    if receipt["error"] != (
        "AssertionError: native MXFP8 numerical gate failed; no model updates permitted"
    ):
        raise ValueError("native failed for a reason other than strict numeric parity")
    if [c["length"] for c in receipt["cases"]] != [3, 31, 33, 127, 129, 257]:
        raise ValueError("incomplete native suite")
    for case in receipt["cases"]:
        if set(case["errors"]) != {"forward", "dq", "dk", "dv"}:
            raise ValueError("incomplete native derivative checks")
        for name, e in case["errors"].items():
            limit = (
                config["native_learning_forward_limit"]
                if name == "forward"
                else config["native_learning_gradient_limit"]
            )
            if (
                not e["finite"]
                or e["relative_l2"] is None
                or not math.isfinite(e["relative_l2"])
                or e["relative_l2"] > limit
            ):
                raise ValueError(
                    "native exceeds the explicit learning comparison envelope"
                )
    if (
        len(receipt["quantizer_layout"]) != 24
        or not all(x["passed"] for x in receipt["quantizer_layout"])
        or receipt["causal_quantization"][
            "across_block_future_perturbation_max_absolute"
        ]
        != 0
    ):
        raise ValueError("native quantization/mask correctness failed")
    return {
        "strict_passed": receipt["passed"],
        "accepted_for_learning_comparison": True,
        "authority": config["learning_acceptance_authority"],
        "forward_limit": config["native_learning_forward_limit"],
        "gradient_limit": config["native_learning_gradient_limit"],
        "causal_quantization": receipt["causal_quantization"],
    }


def job_for(config: dict, source: dict, recipe: dict, backend: str) -> dict:
    """Change only the scoped attention backend and its required fresh gates."""
    if backend not in {"nvidia_mxfp8", "flash_attention_4"}:
        raise ValueError("unsupported MXFP8 screen backend")
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
        "packed_attention_version": "1.31.0"
        if backend == "nvidia_mxfp8"
        else "4.0.0b33",
        "packing_learning_gradient_tolerance": config[
            "candidate_learning_gradient_tolerance"
        ],
    }
    job.pop("startup_validation_reference", None)
    if backend == "flash_attention_4":
        job["startup_validation_reference"] = str(ROOT / config["validation_reference"])
        job["startup_validation_reference_sha256"] = config[
            "validation_reference_sha256"
        ]
    return job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="experiments/b200_nvidia_mxfp8/training_config.yaml"
    )
    config_path = ROOT / parser.parse_args().config
    config = yaml.safe_load(config_path.read_text())
    native_path = ROOT / config["native_reference"]
    acceptance = accept_native(json.loads(native_path.read_text()), config)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    source_paths = [
        config_path,
        Path(__file__),
        ROOT / "experiments/b200_nvidia_mxfp8/run.py",
        ROOT / "src/gleipnir/nvidia_mxfp8_attention.py",
        ROOT / "src/gleipnir/packed_sequences.py",
        ROOT / "src/gleipnir/packed_training.py",
        ROOT / "src/gleipnir/packed_training_screen.py",
        ROOT / config["profile"],
    ]
    source_hashes = {}
    for source_path in source_paths:
        relative = source_path.relative_to(ROOT)
        destination = output / "executed_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        source_hashes[str(relative)] = sha256_file(source_path)
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
    recipe = yaml.safe_load((ROOT / config["profile"]).read_text())["recipe"]
    env = environment(config)
    report = {
        "status": "running",
        "config": config,
        "native_acceptance": acceptance,
        "native_reference_sha256": sha256_file(native_path),
        "conditions": {},
        "source_job": source,
        "executed_source_sha256": source_hashes,
        "cache_paths": {k: v for k, v in env.items() if "CACHE" in k},
    }

    def publish():
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    publish()
    try:
        for backend in config["conditions"]:
            job = job_for(config, source, recipe, backend)
            command = training_command(job) + [
                f"student.init_adapter={ROOT / config['initial_adapter']}",
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
            with (logs / f"{backend}.log").open("x") as handle:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            metadata = json.loads(
                (Path(job["causal_adapter_dir"]) / "training_metadata.json").read_text()
            )
            metrics = summarize(metadata, config["warmup_steps"], accept_learning=True)
            if (
                metrics["initial_master_sha256"]
                != config["expected_initial_master_sha256"]
            ):
                raise ValueError("initial master drift")
            report["conditions"][backend].update(
                status="complete", wall_seconds=time.perf_counter() - started, **metrics
            )
            publish()
            print(
                f"condition_complete={backend} "
                f"mean_seconds={metrics['measured_mean_seconds']}",
                flush=True,
            )
        control, candidate = [
            report["conditions"][k] for k in ["flash_attention_4", "nvidia_mxfp8"]
        ]
        if candidate["physical_contract"] != control["physical_contract"]:
            raise ValueError("physical partitions or tokens do not match")
        gain = (
            1 - candidate["measured_total_seconds"] / control["measured_total_seconds"]
        )
        report.update(
            status="complete",
            measured_time_reduction=gain,
            promising=gain >= config["minimum_relative_improvement"],
            default_promoted=False,
        )
        publish()
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        publish()
        raise


if __name__ == "__main__":
    main()
