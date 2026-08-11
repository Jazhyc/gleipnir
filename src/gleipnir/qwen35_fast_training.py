"""Reproducible fast-kernel setup for Qwen3.5 training."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

FLA_VERSION = "0.5.2"
FLA_PACKAGES = (
    f"flash-linear-attention=={FLA_VERSION}",
    f"fla-core=={FLA_VERSION}",
)
DEFAULT_FLA_TARGET = Path("/tmp/gleipnir-qwen35-fla")
FLA_PROBE = """
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from transformers.utils.import_utils import is_flash_linear_attention_available

if not is_flash_linear_attention_available():
    raise RuntimeError("Transformers did not recognize Flash Linear Attention")
print(
    "qwen_fast_kernel="
    f"{chunk_gated_delta_rule.__module__} "
    f"recurrent_kernel={fused_recurrent_gated_delta_rule.__module__}",
    flush=True,
)
""".strip()


def fla_environment(
    target: Path,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment that imports an isolated pinned FLA install."""
    environment = dict(os.environ if base is None else base)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{target.resolve()}:{existing}" if existing else target.resolve().as_posix()
    )
    environment["GLEIPNIR_FLA_VERSION"] = FLA_VERSION
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return environment


def fla_install_command(
    target: Path,
    *,
    python: Path = Path(sys.executable),
    uv_executable: str = "uv",
) -> list[str]:
    """Build the deterministic isolated-install command used on compute nodes."""
    return [
        uv_executable,
        "pip",
        "install",
        "--python",
        python.absolute().as_posix(),
        "--target",
        target.resolve().as_posix(),
        *FLA_PACKAGES,
        "--no-deps",
    ]


def ensure_fla_kernels(
    target: Path = DEFAULT_FLA_TARGET,
    *,
    python: Path = Path(sys.executable),
) -> dict[str, str]:
    """Install, activate, and probe the pinned Qwen gated-delta kernels."""
    target = target.resolve()
    if not (target / "fla" / "ops").is_dir():
        uv_executable = shutil.which("uv")
        if uv_executable is None:
            user_uv = Path.home() / ".local" / "bin" / "uv"
            uv_executable = user_uv.as_posix() if user_uv.is_file() else None
        if uv_executable is None:
            raise FileNotFoundError(
                "uv is required on PATH or at ~/.local/bin/uv to install "
                "pinned FLA kernels"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            fla_install_command(
                target,
                python=python,
                uv_executable=uv_executable,
            ),
            check=True,
        )
    environment = fla_environment(target)
    subprocess.run(
        [python.absolute().as_posix(), "-c", FLA_PROBE],
        check=True,
        env=environment,
    )
    return environment
