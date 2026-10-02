"""Pinned, isolated runtime for packed monitoring training and reference canaries."""

from __future__ import annotations

import os
from pathlib import Path

from gleipnir.flashqla_training import flashqla_environment
from gleipnir.monitoring_systems_screen import gpu_environment
from gleipnir.qwen35_fast_training import (
    DEFAULT_CAUSAL_CONV1D_TARGET,
    DEFAULT_FLA_TARGET,
    DEFAULT_TRITON_TARGET,
    causal_conv1d_environment,
    fla_environment,
    triton_environment,
)


def training_environment(root: Path, cache_name: str) -> dict[str, str]:
    env = flashqla_environment(
        triton_environment(
            DEFAULT_TRITON_TARGET,
            causal_conv1d_environment(
                DEFAULT_CAUSAL_CONV1D_TARGET,
                fla_environment(DEFAULT_FLA_TARGET, dict(os.environ)),
            ),
        )
    )
    env = gpu_environment(env, 0, root / ".cache/training" / cache_name)
    env.update(
        FLA_DISABLE_BACKEND_DISPATCH="1",
        OMP_NUM_THREADS="4",
        WANDB_MODE="disabled",
        PYTHONUNBUFFERED="1",
    )
    return env
