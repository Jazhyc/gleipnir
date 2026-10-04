"""Pinned, isolated runtime for packed monitoring training and reference canaries."""

from __future__ import annotations

import os
from pathlib import Path

from gleipnir.attention_backends import packed_fa4_environment
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


def training_cache(
    root: Path, cache_name: str, *, isolated_cache: bool = False
) -> Path:
    """Reuse the populated persistent cache unless cold-cache isolation is explicit."""
    parent = root / ".cache/training"
    if isolated_cache:
        return parent / cache_name
    shared = parent / "shared"
    parent.mkdir(parents=True, exist_ok=True)
    if not shared.exists():
        legacy = parent / "student_injection_awareness"
        try:
            if legacy.is_dir():
                shared.symlink_to(legacy.name, target_is_directory=True)
            else:
                shared.mkdir()
        except FileExistsError:
            pass  # Another launcher may have initialized the shared cache first.
    return shared.resolve()


def training_environment(
    root: Path, cache_name: str, *, isolated_cache: bool = False
) -> dict[str, str]:
    env = flashqla_environment(
        triton_environment(
            DEFAULT_TRITON_TARGET,
            causal_conv1d_environment(
                DEFAULT_CAUSAL_CONV1D_TARGET,
                fla_environment(DEFAULT_FLA_TARGET, dict(os.environ)),
            ),
        )
    )
    env = gpu_environment(
        env, 0, training_cache(root, cache_name, isolated_cache=isolated_cache)
    )
    env = packed_fa4_environment(env, root)
    env.update(
        FLA_DISABLE_BACKEND_DISPATCH="1",
        OMP_NUM_THREADS="4",
        WANDB_MODE="disabled",
        PYTHONUNBUFFERED="1",
    )
    return env
