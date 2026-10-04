"""Sequential references and per-backbone persistent engines on one GPU."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from experiments.augmented_judge_evaluation.prepare import ROOT
from gleipnir.qwen35_fast_training import ensure_qwen35_long_trajectory_kernels


def main() -> None:
    env = ensure_qwen35_long_trajectory_kernels(
        ROOT / ".cache/kernels/fla-0.5.2",
        ROOT / ".cache/kernels/causal-conv1d-1.6.2.post1",
        ROOT / ".cache/kernels/triton-3.7.1",
        python=Path(sys.executable),
    )
    env.update(FLA_DISABLE_BACKEND_DISPATCH="1", PYTHONUNBUFFERED="1")
    command = [sys.executable, "-m", "experiments.augmented_judge_evaluation.evaluate"]
    for size in ("4b", "9b"):
        subprocess.run(
            [*command, "--stage", "reference", "--size", size],
            cwd=ROOT,
            env=env,
            check=True,
        )
        subprocess.run(
            [*command, "--stage", "vllm", "--size", size],
            cwd=ROOT,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
            check=True,
        )
    subprocess.run([*command, "--stage", "report"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
