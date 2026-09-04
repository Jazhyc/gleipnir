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
CAUSAL_CONV1D_VERSION = "1.6.2.post1"
CAUSAL_CONV1D_PACKAGE = f"causal-conv1d=={CAUSAL_CONV1D_VERSION}"
DEFAULT_CAUSAL_CONV1D_TARGET = Path("/tmp/gleipnir-qwen35-causal-conv1d")
DEFAULT_CUDA_HOME = Path("/usr/local/cuda")
TRITON_VERSION = "3.7.1"
TRITON_PACKAGE = f"triton=={TRITON_VERSION}"
DEFAULT_TRITON_TARGET = Path("/tmp/gleipnir-triton-3.7.1")
FLA_PROBE = """
import fla
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from transformers.utils.import_utils import is_flash_linear_attention_available

if fla.__version__ != "0.5.2":
    raise RuntimeError(f"expected FLA 0.5.2, found {fla.__version__}")
if not is_flash_linear_attention_available():
    raise RuntimeError("Transformers did not recognize Flash Linear Attention")
print(
    "qwen_fast_kernel="
    f"{chunk_gated_delta_rule.__module__} "
    f"recurrent_kernel={fused_recurrent_gated_delta_rule.__module__}",
    flush=True,
)
""".strip()
CAUSAL_CONV1D_PROBE = """
import causal_conv1d
import torch
from causal_conv1d import causal_conv1d_fn
from transformers.utils.import_utils import is_causal_conv1d_available

if causal_conv1d.__version__ != "1.6.2.post1":
    raise RuntimeError(
        f"expected causal-conv1d 1.6.2.post1, found {causal_conv1d.__version__}"
    )
if not is_causal_conv1d_available():
    raise RuntimeError("Transformers did not recognize causal-conv1d")
if not torch.cuda.is_available():
    raise RuntimeError("causal-conv1d preflight requires a CUDA device")
x = torch.randn(1, 128, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
weight = torch.randn(128, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
output = causal_conv1d_fn(x=x, weight=weight, activation="silu")
output.float().square().mean().backward()
torch.cuda.synchronize()
print(
    "qwen_causal_conv1d_kernel="
    f"{causal_conv1d_fn.__module__} version={causal_conv1d.__version__}",
    flush=True,
)
""".strip()
TRITON_PROBE = """
import triton

if triton.__version__ != "3.7.1":
    raise RuntimeError(f"expected Triton 3.7.1, found {triton.__version__}")
print(f"qwen_triton_version={triton.__version__}", flush=True)
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
    if DEFAULT_CUDA_HOME.is_dir():
        environment.setdefault("CUDA_HOME", DEFAULT_CUDA_HOME.as_posix())
    return environment


def causal_conv1d_environment(
    target: Path,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment that imports an isolated causal-conv1d build."""
    environment = dict(os.environ if base is None else base)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{target.resolve()}:{existing}" if existing else target.resolve().as_posix()
    )
    environment["GLEIPNIR_CAUSAL_CONV1D_VERSION"] = CAUSAL_CONV1D_VERSION
    if DEFAULT_CUDA_HOME.is_dir():
        environment.setdefault("CUDA_HOME", DEFAULT_CUDA_HOME.as_posix())
    return environment


def triton_environment(
    target: Path,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment that imports an isolated pinned Triton build."""
    environment = dict(os.environ if base is None else base)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{target.resolve()}:{existing}" if existing else target.resolve().as_posix()
    )
    environment["GLEIPNIR_TRITON_VERSION"] = TRITON_VERSION
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


def causal_conv1d_install_command(
    target: Path,
    *,
    python: Path = Path(sys.executable),
    uv_executable: str = "uv",
) -> list[str]:
    """Build the isolated source-install command for causal-conv1d."""
    return [
        uv_executable,
        "pip",
        "install",
        "--python",
        python.absolute().as_posix(),
        "--target",
        target.resolve().as_posix(),
        "--no-deps",
        "--no-build-isolation",
        CAUSAL_CONV1D_PACKAGE,
    ]


def triton_install_command(
    target: Path,
    *,
    python: Path = Path(sys.executable),
    uv_executable: str = "uv",
) -> list[str]:
    """Return the deterministic isolated Triton install command."""
    return [
        uv_executable,
        "pip",
        "install",
        "--python",
        python.absolute().as_posix(),
        "--target",
        target.resolve().as_posix(),
        "--no-deps",
        TRITON_PACKAGE,
    ]


def _find_uv() -> str:
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        user_uv = Path.home() / ".local" / "bin" / "uv"
        uv_executable = user_uv.as_posix() if user_uv.is_file() else None
    if uv_executable is None:
        raise FileNotFoundError(
            "uv is required on PATH or at ~/.local/bin/uv to install fast kernels"
        )
    return uv_executable


def ensure_fla_kernels(
    target: Path = DEFAULT_FLA_TARGET,
    *,
    python: Path = Path(sys.executable),
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Install, activate, and probe the pinned Qwen gated-delta kernels."""
    target = target.resolve()
    if not (target / f"flash_linear_attention-{FLA_VERSION}.dist-info").is_dir():
        uv_executable = _find_uv()
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            fla_install_command(
                target,
                python=python,
                uv_executable=uv_executable,
            ),
            check=True,
        )
    environment = fla_environment(target, base)
    subprocess.run(
        [python.absolute().as_posix(), "-c", FLA_PROBE],
        check=True,
        env=environment,
    )
    return environment


def ensure_causal_conv1d(
    target: Path = DEFAULT_CAUSAL_CONV1D_TARGET,
    *,
    python: Path = Path(sys.executable),
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build, activate, and execute a BF16 causal-conv1d GPU preflight."""
    target = target.resolve()
    environment = causal_conv1d_environment(target, base)
    if not (target / f"causal_conv1d-{CAUSAL_CONV1D_VERSION}.dist-info").is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        build_environment = dict(environment)
        build_environment["CAUSAL_CONV1D_FORCE_BUILD"] = "TRUE"
        build_environment.setdefault("MAX_JOBS", "4")
        subprocess.run(
            causal_conv1d_install_command(
                target,
                python=python,
                uv_executable=_find_uv(),
            ),
            check=True,
            env=build_environment,
        )
    subprocess.run(
        [python.absolute().as_posix(), "-c", CAUSAL_CONV1D_PROBE],
        check=True,
        env=environment,
    )
    return environment


def ensure_qwen35_fast_kernels(
    fla_target: Path = DEFAULT_FLA_TARGET,
    causal_conv1d_target: Path = DEFAULT_CAUSAL_CONV1D_TARGET,
    *,
    python: Path = Path(sys.executable),
) -> dict[str, str]:
    """Fail closed unless both Qwen3.5 training fast paths work on the GPU."""
    environment = ensure_fla_kernels(fla_target, python=python)
    return ensure_causal_conv1d(
        causal_conv1d_target,
        python=python,
        base=environment,
    )


def ensure_qwen35_long_trajectory_kernels(
    fla_target: Path = DEFAULT_FLA_TARGET,
    causal_conv1d_target: Path = DEFAULT_CAUSAL_CONV1D_TARGET,
    triton_target: Path = DEFAULT_TRITON_TARGET,
    *,
    python: Path = Path(sys.executable),
) -> dict[str, str]:
    """Activate FLA, causal-conv1d, and Hopper-safe Triton for long training."""
    environment = ensure_qwen35_fast_kernels(
        fla_target,
        causal_conv1d_target,
        python=python,
    )
    target = triton_target.resolve()
    if not (target / f"triton-{TRITON_VERSION}.dist-info").is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            triton_install_command(
                target,
                python=python,
                uv_executable=_find_uv(),
            ),
            check=True,
        )
    environment = triton_environment(target, environment)
    subprocess.run(
        [python.absolute().as_posix(), "-c", TRITON_PROBE],
        check=True,
        env=environment,
    )
    return environment


def prewarm_gated_delta_rule(sequence_length: int) -> dict[str, int | str]:
    """Autotune Qwen3.5's FLA forward/backward before loading model weights."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("FLA prewarm requires CUDA")

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tensor_shape = (1, sequence_length, 32, 128)
    gate_shape = (1, sequence_length, 32)
    query = torch.randn(tensor_shape, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(tensor_shape, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(tensor_shape, device=device, dtype=dtype, requires_grad=True)
    gate = -torch.rand(gate_shape, device=device, dtype=torch.float32)
    gate.requires_grad_(True)
    beta = torch.rand(gate_shape, device=device, dtype=dtype, requires_grad=True)

    torch.cuda.reset_peak_memory_stats(device)
    output, _ = chunk_gated_delta_rule(
        query,
        key,
        value,
        g=gate,
        beta=beta,
        use_qk_l2norm_in_kernel=True,
    )
    output[..., 0, 0].float().mean().backward()
    torch.cuda.synchronize(device)
    result: dict[str, int | str] = {
        "device": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "sequence_length": sequence_length,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "status": "passed",
    }
    print(f"fla_in_process_prewarm={result}", flush=True)
    return result
