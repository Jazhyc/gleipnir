"""Reconstruct an existing score runtime from public, pinned serving receipts."""

from __future__ import annotations

import json
from pathlib import Path


def resume_score_environment(
    root: Path, parent: dict, base_environment: dict[str, str]
) -> dict[str, str]:
    """Reuse the staged runtime and compiler caches without serializing secrets."""
    from gleipnir.serving.gigatoken import configure_frontend
    from gleipnir.serving.runtime import local_serving_runtime
    from gleipnir.training.backends.native_fp4 import native_fp4_environment
    from gleipnir.training.backends.qwen35 import (
        DEFAULT_TRITON_TARGET,
        triton_environment,
    )

    env = native_fp4_environment(
        triton_environment(DEFAULT_TRITON_TARGET, base_environment.copy()), root
    )
    env["PYTHONPATH"] += f":{root / '.cache/kernels/fa4'}"
    runtime = local_serving_runtime(root, env)
    if (
        runtime is None
        or runtime["manifest_sha256"] != parent["local_runtime"]["manifest_sha256"]
        or runtime["python"] != parent["command"][0]
    ):
        raise ValueError("retired-parent staged runtime identity changed")
    frontend_command = parent["command"].copy()
    frontend_command[frontend_command.index("-m") + 1] = (
        "experiments.b200_attention_gdn_serving.server"
    )
    configure_frontend(root, parent["frontend"], frontend_command, env)
    env["GLEIPNIR_FROST_WRAPPER_VALIDATION"] = parent["host_wrapper"]["validation"]
    env["VLLM_SERVER_DEV_MODE"] = "1"
    selection = json.loads(
        (root / "experiments/b200_inference_benchmark/baseline.json").read_text()
    )
    env["GLEIPNIR_STRIDE_VALIDATION"] = selection["mutation_validation"]
    for key, value in parent["cache_paths"].items():
        if key in {
            "HF_HUB_CACHE",
            "VLLM_CACHE_ROOT",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "TILELANG_CACHE_DIR",
            "TVM_CACHE_DIR",
            "CUDNN_FRONTEND_COMPILED_CACHE",
            "CUDNN_FRONTEND_COMPILED_CACHE_MAX_BYTES",
            "CUTE_DSL_CACHE_DIR",
            "FLASHINFER_CACHE_DIR",
        }:
            env[key] = value
    env["FLASHINFER_WORKSPACE_BASE"] = str(root / ".cache/flashinfer")
    return env
