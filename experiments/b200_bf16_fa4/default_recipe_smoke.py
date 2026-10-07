"""Check two ordinary FA4-default updates while reusing the completed gates."""

import json
import subprocess
from pathlib import Path

import yaml

from gleipnir.monitoring_campaign_runtime import training_environment
from gleipnir.monitoring_systems_screen import sha256_file
from gleipnir.monitoring_training_command import training_command

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    baseline = json.loads(
        (ROOT / "results/b200_bf16_fa4_accepted/summary.json").read_text()
    )
    profile = ROOT / "src/gleipnir/configs/systems_screen/qwen35_4b_b200_bf16_fa4.yaml"
    recipe = yaml.safe_load(profile.read_text())["recipe"]
    output = ROOT / "results/b200_fa4_default_smoke"
    logs = ROOT / "logs/runpod/b200_fa4_default"
    output.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=True)
    job = {
        **baseline["conditions"]["sdpa"]["job"],
        **recipe,
        "job_name": "fa4-default-smoke",
        "max_steps": 2,
        "save_steps": 1000000,
        "expected_initial_master_sha256": baseline["conditions"]["sdpa"][
            "initial_master_sha256"
        ],
        "output_dir": str(output),
        "causal_adapter_dir": str(output / "causal_adapter"),
        "model_dir": str(output / "model"),
        "hydra_log_dir": str(logs),
    }
    for path_key, hash_key in [
        ("student_rows", "student_rows_sha256"),
        ("soft_targets", "soft_targets_sha256"),
        ("selection_manifest", "selection_sha256"),
    ]:
        if sha256_file(Path(job[path_key])) != job[hash_key]:
            raise ValueError(f"input drift: {path_key}")
    initial = ROOT / "results/fp4_row_aot_training/initial_adapter"
    for name, digest in baseline["initial_adapter_files"].items():
        if sha256_file(initial / name) != digest:
            raise ValueError(f"initial adapter drift: {name}")
    command = training_command(job) + [f"student.init_adapter={initial}"]
    environment = training_environment(ROOT, "b200_fa4_default")
    files = [
        Path(__file__).relative_to(ROOT),
        profile.relative_to(ROOT),
        Path("src/gleipnir/validated_startup.py"),
        Path("src/gleipnir/training/packed.py"),
        Path("src/gleipnir/__init__.py"),
        Path("src/gleipnir/_compat.py"),
        Path("src/gleipnir/campaigns/training_command.py"),
        Path("src/gleipnir/monitoring_campaign_runtime.py"),
        Path("src/gleipnir/attention_backends.py"),
        Path("experiments/deception_distillation/train_student_sft.py"),
    ]
    contract = {
        "job": job,
        "command": command,
        "source_sha256": {str(p): sha256_file(ROOT / p) for p in files},
        "initial_adapter_files": baseline["initial_adapter_files"],
        "cache_paths": {k: v for k, v in environment.items() if "CACHE" in k},
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    for path in files:
        archive = output / "executed_sources" / path
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes((ROOT / path).read_bytes())
    print("fa4_default_smoke_start", flush=True)
    with (logs / "training.log").open("w") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    metadata = json.loads(
        (output / "causal_adapter/training_metadata.json").read_text()
    )
    packing = metadata["sequence_packing"]
    if (
        metadata["training_state"]["global_step"] != 2
        or metadata["checkpointed_layer_indices"]
        or not metadata["quantization"]["full_bf16_lora"]["verified"]
        or packing["initial_master_sha256"] == packing["final_master_sha256"]
        or packing["attention_backend"] != "flash_attention_4"
        or metadata["startup_validation"]["reference_sha256"]
        != recipe["startup_validation_reference_sha256"]
    ):
        raise ValueError("default training contract failed")
    for name in ["eager_canary", "compiled_canary", "preflight"]:
        if (
            packing[name].get("performed_this_run") is not False
            or "passed" in packing[name]
        ):
            raise ValueError("reused checks were mislabeled")
    records = metadata["adaptive_microbatching"]["records"]
    if (
        any(r["tokens"] != r["padded_tokens"] for r in records)
        or not metadata["adaptive_microbatching"]["require_finite_gradients"]
    ):
        raise ValueError("packing/finite-gradient contract failed")
    summary = {
        "status": "complete",
        "steps": 2,
        "master_changed": True,
        "physical_calls": len(records),
        "startup_validation": metadata["startup_validation"],
        "packing": packing,
        "peak_allocated_gib": metadata["peak_cuda_memory_allocated_bytes"] / 2**30,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("fa4_default_smoke_complete", flush=True)


if __name__ == "__main__":
    main()
