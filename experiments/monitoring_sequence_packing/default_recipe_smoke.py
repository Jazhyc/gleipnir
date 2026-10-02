"""Exercise the selected packed recipe through the ordinary Trainer entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml

from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)
from gleipnir.flashqla_training import flashqla_environment
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


def smoke_job(config: dict, source: dict, recipe: dict) -> dict:
    """Keep the source cohort fixed while exercising ordinary packed training."""
    if config["steps"] != 2 or not recipe.get("sequence_packing"):
        raise ValueError("default smoke requires two packed optimizer steps")
    output = ROOT / config["output"]
    return {
        **source,
        **recipe,
        "job_name": "packed-default-smoke",
        "max_steps": 2,
        "save_steps": 1000000,
        "output_dir": str(output),
        "causal_adapter_dir": str(output / "causal_adapter"),
        "model_dir": str(output / "model"),
        "hydra_log_dir": str(ROOT / config["logs"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "experiments/monitoring_sequence_packing/default_recipe_smoke.yaml"
        ),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    sources = [
        json.loads(line)
        for line in (ROOT / config["source_jobs"]).read_text().splitlines()
        if line
    ]
    (source,) = [s for s in sources if s["job_name"] == config["condition_source"]]
    profile = yaml.safe_load((ROOT / config["profile"]).read_text())
    job = smoke_job(config, source, profile["recipe"])
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    initial = ROOT / config["initial_adapter"]
    command = training_command(job) + [f"student.init_adapter={initial}"]
    files = [
        config["profile"],
        str(args.config),
        str(Path(__file__).relative_to(ROOT)),
        "experiments/deception_distillation/train_student_sft.py",
        "experiments/tool_trajectory_monitoring/run_distillation_train.py",
        "src/gleipnir/packed_training.py",
        "src/gleipnir/packed_sequences.py",
        "src/gleipnir/packed_training_screen.py",
        "src/gleipnir/adaptive_microbatching.py",
        "src/gleipnir/flashqla_training.py",
        "src/gleipnir/bf16_lora.py",
    ]
    contract = {
        "config": config,
        "job": job,
        "command": command,
        "source_sha256": {p: sha256_file(ROOT / p) for p in files},
        "initial_adapter_sha256": {
            p.name: sha256_file(p) for p in initial.iterdir() if p.is_file()
        },
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    for path_key, hash_key in [
        ("student_rows", "student_rows_sha256"),
        ("soft_targets", "soft_targets_sha256"),
        ("selection_manifest", "selection_sha256"),
    ]:
        if sha256_file(Path(job[path_key])) != job[hash_key]:
            raise ValueError(f"smoke input hash mismatch: {path_key}")
    environment = flashqla_environment(
        triton_environment(
            DEFAULT_TRITON_TARGET,
            causal_conv1d_environment(
                DEFAULT_CAUSAL_CONV1D_TARGET,
                fla_environment(DEFAULT_FLA_TARGET, dict(os.environ)),
            ),
        )
    )
    environment = gpu_environment(
        environment, 0, ROOT / ".cache/training/qwen35_4b_b200_fa4/gpu-0"
    )
    environment["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
    environment["OMP_NUM_THREADS"] = "4"
    print("ordinary_packed_training_start", flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    metadata = json.loads(
        (Path(job["causal_adapter_dir"]) / "training_metadata.json").read_text()
    )
    packing = metadata["sequence_packing"]
    if packing["initial_master_sha256"] != config["expected_initial_master_sha256"]:
        raise ValueError("smoke initial master mismatch")
    if (
        metadata["training_state"]["global_step"] != 2
        or metadata["checkpointed_layer_indices"]
        or not metadata["quantization"]["full_bf16_lora"]["verified"]
        or not all(
            packing[key]["passed"]
            for key in ["eager_canary", "compiled_canary", "preflight"]
        )
        or packing["initial_master_sha256"] == packing["final_master_sha256"]
    ):
        raise ValueError("ordinary packed smoke gates failed")
    records = metadata["adaptive_microbatching"]["records"]
    if any(row["padded_tokens"] != row["tokens"] for row in records):
        raise ValueError("ordinary packing introduced padding")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "steps": 2,
                "fp32_master_changed": True,
                "packing": packing,
                "physical_calls": len(records),
                "logical_batch_sizes": metadata["adaptive_microbatching"][
                    "logical_batch_sizes"
                ],
                "peak_allocated_gib": metadata["peak_cuda_memory_allocated_bytes"]
                / 2**30,
            },
            indent=2,
        )
        + "\n"
    )
    print("ordinary_packed_training_complete", flush=True)


if __name__ == "__main__":
    main()
