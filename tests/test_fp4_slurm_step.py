"""The shared entrypoint rejects invalid launches before any remote/GPU work."""

import os
import subprocess


def test_step_requires_existing_allocation():
    environment = dict(os.environ, SLURM_JOB_ID="")
    result = subprocess.run(
        ["bash", "cluster/slurm/fp4_inference_step.sh", "baseline"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "existing Slurm allocation" in result.stderr


def test_step_rejects_unrelated_module_before_environment_setup():
    environment = dict(os.environ, SLURM_JOB_ID="123")
    result = subprocess.run(
        ["bash", "cluster/slurm/fp4_inference_step.sh", "--module", "os"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Unsupported" in result.stderr
