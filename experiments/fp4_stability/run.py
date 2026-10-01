"""Run bounded native FP4 forward diagnostics or precision trajectories."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from experiments.b200_fouroversix.run import make_job, verify_inputs
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


def validate_config(config: dict) -> None:
    """Fail before GPU model loading for unsupported or unbounded campaigns."""
    if config["steps"] not in {1, 10}:
        raise ValueError("retain one preflight update or ten matched updates")
    if config["diagnostics_only"] and config["steps"] != 1:
        raise ValueError("forward diagnostics must use the global preflight selection")
    if not config["conditions"] or any(
        name not in {"nf4", "bf16", "fouroversix"} for name in config["conditions"]
    ):
        raise ValueError("unknown or empty precision conditions")
    if config["backward_mode"] not in {"fp4", "dequantized_bf16"}:
        raise ValueError("unknown backward precision")
    if config["compile_backend"] not in {"inductor", "aot_eager", "eager"}:
        raise ValueError("unknown diagnostic compiler backend")


def campaign_stages(config: dict, source: dict, global_longest: dict) -> list[dict]:
    """Require an actual global-longest update before every ten-update condition."""
    stages = []
    for precision in config["conditions"]:
        if config["steps"] == 10:
            stages.append(
                dict(
                    name=f"{precision}-global-preflight",
                    precision=precision,
                    source=global_longest,
                    steps=1,
                )
            )
        stages.append(
            dict(
                name=precision,
                precision=precision,
                source=source,
                steps=config["steps"],
            )
        )
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    validate_config(config)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    logs = ROOT / config["logs"]
    logs.mkdir(parents=True, exist_ok=True)
    source_path = config["preflight_jobs"] if config["steps"] == 1 else config["jobs"]
    jobs = [json.loads(line) for line in (ROOT / source_path).read_text().splitlines()]
    source = (
        jobs[0]
        if config["steps"] == 1
        else next(job for job in jobs if job["job_name"] == config["condition_source"])
    )
    verify_inputs(source)
    global_jobs = [
        json.loads(line)
        for line in (ROOT / config["preflight_jobs"]).read_text().splitlines()
        if line
    ]
    if len(global_jobs) != 1:
        raise ValueError("expected the frozen global-longest-32 selection")
    verify_inputs(global_jobs[0])
    environment = triton_environment(
        DEFAULT_TRITON_TARGET,
        causal_conv1d_environment(
            DEFAULT_CAUSAL_CONV1D_TARGET, fla_environment(DEFAULT_FLA_TARGET)
        ),
    )
    environment["PYTHONPATH"] = (
        f"{ROOT / config['kernel_target']}:{environment['PYTHONPATH']}"
    )
    environment["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
    environment["OMP_NUM_THREADS"] = "4"
    environment["TORCHINDUCTOR_EMULATE_PRECISION_CASTS"] = (
        "1" if config.get("emulate_precision_casts", False) else "0"
    )
    cache = ROOT / config["compiler_cache"]
    cache.mkdir(parents=True, exist_ok=True)
    environment = gpu_environment(environment, 0, cache)
    contract = {
        "config": config,
        "source_job": source,
        "global_longest_job": global_jobs[0],
        "inputs_verified": True,
        "inductor_emulate_precision_casts": config.get(
            "emulate_precision_casts", False
        ),
        "source_revision": os.environ.get("GLEIPNIR_COMMIT"),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [
                ROOT / "src/gleipnir/fouroversix_training.py",
                ROOT / "src/gleipnir/precision_training_screen.py",
                ROOT / "experiments/deception_distillation/train_student_sft.py",
                Path(__file__),
                args.config.resolve(),
            ]
        },
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    status = {"status": "running", "stages": []}

    def publish() -> None:
        (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")

    publish()
    try:
        canary_command = [
            sys.executable,
            "experiments/b200_fouroversix/kernel_canary.py",
            "--output",
            str(output / "kernel_canary.json"),
            "--backward-mode",
            config["backward_mode"],
        ]
        if config.get("row_scaled_activations", False):
            canary_command.append("--row-scaled-activations")
        with (logs / "kernel-canary.log").open("w") as handle:
            subprocess.run(
                canary_command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        for specification in campaign_stages(config, source, global_jobs[0]):
            precision = specification["precision"]
            stage_steps = specification["steps"]
            destination = output / specification["name"]
            job = make_job(specification["source"], destination, precision, stage_steps)
            command = training_command(job) + [
                f"++student.quantization.mlp_precision={precision}",
                f"++student.quantization.fp4_backward_mode={config['backward_mode']}",
                "++student.quantization.fp4_row_scaled_activations="
                f"{str(config.get('row_scaled_activations', False)).lower()}",
                "++student.training.adapter_init_seed=0",
                "++student.training.precision_screen.enabled=true",
                f"++student.training.precision_screen.output_dir={destination}",
                f"++student.training.precision_screen.steps={stage_steps}",
                f"++student.training.precision_screen.diagnostics_only={str(config['diagnostics_only']).lower()}",
                f"student.training.selective_torch_compile_backend={config['compile_backend']}",
            ]
            if stage_steps == 1:
                command.append("student.training.warmup_ratio=0.0")
            stage = {
                "name": specification["name"],
                "precision": precision,
                "steps": stage_steps,
                "selection_manifest": specification["source"]["selection_manifest"],
                "command": command,
                "status": "running",
                "started_unix": time.time(),
            }
            status["stages"].append(stage)
            publish()
            print(f"starting_stage={specification['name']}", flush=True)
            with (logs / f"{specification['name']}.log").open("w") as handle:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            stage.update(
                returncode=result.returncode,
                ended_unix=time.time(),
                status="complete" if result.returncode == 0 else "failed",
            )
            publish()
            if result.returncode:
                raise RuntimeError(f"{specification['name']} failed; inspect {logs}")
        status["status"] = "complete"
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        publish()


if __name__ == "__main__":
    main()
