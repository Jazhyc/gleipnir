"""Construct the isolated candidate environment without changing old runtimes."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def candidate_environment(root: Path) -> dict[str, str]:
    """Reuse audited native dependencies and durable compiler caches."""
    env = os.environ.copy()
    native = Path("/tmp/gleipnir-serving-runtime")
    paths = [native / "fa4", root / "src", root, native / "mxfp8/frontend"]
    if not all(path.is_dir() for path in paths):
        raise ValueError("preserved native dependency overlays are missing")
    env.update(
        PATH=":".join(
            [
                str(Path(sys.executable).parent),
                "/usr/local/cuda/bin",
                env.get("PATH", ""),
            ]
        ),
        PYTHONPATH=":".join(map(str, paths)),
        LD_LIBRARY_PATH=":".join(
            [
                str(native / "mxfp8/runtime/nvidia/cudnn/lib"),
                env.get("LD_LIBRARY_PATH", ""),
            ]
        ).rstrip(":"),
        GLEIPNIR_NVIDIA_SOURCE=str(native / "mxfp8/source"),
        CUDNN_FRONTEND_ENABLE_FROST_ENGINES="1",
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_SERVER_DEV_MODE="1",
        TOKENIZERS_PARALLELISM="false",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        OMP_NUM_THREADS="4",
        MAX_JOBS="4",
        TORCHINDUCTOR_COMPILE_THREADS="16",
        CUDA_HOME="/usr/local/cuda",
        PYTHONUNBUFFERED="1",
    )
    shared = root / ".cache/training/shared"
    for key, path in {
        "HF_HOME": root / ".cache/huggingface",
        "HF_HUB_CACHE": root / ".cache/huggingface/hub",
        "VLLM_CACHE_ROOT": root / ".cache/vllm/student_injection_awareness_v1",
        "TORCHINDUCTOR_CACHE_DIR": root
        / ".cache/torchinductor/student_injection_awareness_v1",
        "TRITON_CACHE_DIR": shared / "gpu-0/triton",
        "CUDNN_FRONTEND_COMPILED_CACHE": shared / "cudnn_frontend",
        "CUTE_DSL_CACHE_DIR": shared / "cute_dsl",
        "FLASHINFER_WORKSPACE_BASE": root / ".cache/flashinfer",
    }.items():
        env[key] = str(path)
    env["CUDNN_FRONTEND_COMPILED_CACHE_MAX_BYTES"] = "0"
    return env


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    raise SystemExit(
        subprocess.call(
            [sys.executable, *sys.argv[1:]], cwd=root, env=candidate_environment(root)
        )
    )
