"""Freeze and launch matched gradient, optimizer-update, and trajectory checks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml

from experiments.b200_adaptive_microbatching.diagnose import diagnostic_job
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if any(
        cfg[k] != v
        for k, v in {
            "steps": 10,
            "logical_batch_size": 32,
            "max_padded_tokens": 16384,
            "max_microbatch_size": 8,
            "backward_pass_autocast": "same_as_forward",
            "seed": 0,
        }.items()
    ):
        raise ValueError("execution audit requires the predeclared bounded contract")
    jobs_file = ROOT / cfg["jobs"]
    original = next(
        j
        for j in (
            json.loads(line) for line in jobs_file.read_text().splitlines() if line
        )
        if j["job_name"] == cfg["condition_source"]
    )
    for path_key, hash_key in (
        ("student_rows", "student_rows_sha256"),
        ("soft_targets", "soft_targets_sha256"),
        ("selection_manifest", "selection_sha256"),
    ):
        if sha256_file(Path(original[path_key])) != original[hash_key]:
            raise ValueError(f"frozen input checksum changed: {path_key}")
    output = (ROOT / cfg["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    job = diagnostic_job(original, output, "compiled")
    job.update(
        job_name="execution-audit",
        design_role="bounded_matched_execution_learning_diagnostic",
        max_steps=10,
        selective_torch_compile_canary_tokens=0,
    )
    command = training_command(job)
    command.extend(
        [
            "++student.training.execution_audit.enabled=true",
            f"++student.training.execution_audit.output_dir={output}",
            "++student.training.execution_audit.steps=10",
        ]
    )
    command = [
        command[0],
        str(Path(__file__).with_name("autocast_canary.py")),
        cfg["backward_pass_autocast"],
        *command[1:],
    ]
    contract = {
        "config": cfg,
        "config_sha256": sha256_file(args.config),
        "job": job,
        "command": command,
        "revision": os.environ["GLEIPNIR_COMMIT"],
        "original_jobs_sha256": sha256_file(jobs_file),
        "input_checksums_verified": True,
        "new_authorization": (
            "user requested all three checks including ten optimizer steps; "
            "preserve failed timing screen and its gates"
        ),
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    environment = triton_environment(
        DEFAULT_TRITON_TARGET,
        causal_conv1d_environment(
            DEFAULT_CAUSAL_CONV1D_TARGET, fla_environment(DEFAULT_FLA_TARGET)
        ),
    )
    cache = ROOT / ".cache/training/qwen35_4b_b200_fa4/gpu-0"
    if not cache.is_dir():
        raise ValueError("execution audit requires the existing compatible cache")
    subprocess.run(
        command, cwd=ROOT, env=gpu_environment(environment, 0, cache), check=True
    )


if __name__ == "__main__":
    main()
