"""The selected B200 NVFP4 MLP recipe, independent of benchmark entrypoints."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from gleipnir._compat import canonical_source_reference

REFERENCE = "results/b200_mlp_gemm/warmed03/causal_adapter/training_metadata.json"
REFERENCE_SHA256 = "14ab15279bb8895cf32353117d5c1cf957ad45d2b0ca9d7205067db27d77edeb"
HISTORICAL_KERNEL_SHA256 = {
    "cudnn_fp4_mlp.py": (
        "64f968420eec392171cad076116e7a2c00e298f0b08285be0871deed29fd1a93"
    ),
    "cudnn_fp4_gemm.py": (
        "e829be25b924c1252c336c44b525c32aa68ef9a1d44f635e5d4b65675ee05675"
    ),
    "cudnn_fp4_epilogue.py": (
        "fa8060d11c836ba622b82033d9f8321635550eb8c638abc9ec246c55d598cd56"
    ),
    "nvfp4_pack.py": "f359305eb051547b75fc7ef2fbc4409cb857a2fd94733f2dacdbd38d2d01091a",
    "attention_backends.py": (
        "26f6b0edc8c35bf7de04e0087451a1b5d8812c8508a51a444684f56cbf53e1c8"
    ),
}
# Source promotion is based on targeted bitwise equivalence, not a new pass of
# the historical model canaries. Preserve both source generations and receipts.
KERNEL_SHA256 = {
    **HISTORICAL_KERNEL_SHA256,
    "cudnn_fp4_mlp.py": (
        "35022148da1813dc07408826b39d6584efa0c8efe244a5286940354415ab5dd0"
    ),
    "cudnn_fp4_gemm.py": (
        "d6b1d7ca14860ecdf80ea95730df46b603685e8a2d7a00c1fd721d2ad045741e"
    ),
    "cudnn_fp4_epilogue.py": (
        "b06e7131a0834154c1a27d01bf96a9f238208154947b3ce0b5a6e86e1fa742e2"
    ),
}
RUNTIME_SHAPE_VALIDATION = {
    "basis": "targeted_bitwise_runtime_shape_equivalence",
    "performed_this_run": False,
    "historical_source_sha256": HISTORICAL_KERNEL_SHA256,
    "active_source_sha256": KERNEL_SHA256,
    "receipts": {
        "runtime_m_probe": {
            "cases": 16,
            "sha256": (
                "e1ebbb36f1e43d8b68eed98690df38d924c936c98a4e2970d523c2c739eeed80"
            ),
        },
        "runtime_conversion_probe": {
            "cases": 21,
            "sha256": (
                "8fe670c604dc0e44eae9a32b7dfa1dee95c40e37fe6e708f27e4e34bfeacf27a"
            ),
        },
        "runtime_scale_probe": {
            "cases": 7,
            "sha256": (
                "e27d7862f5de2f8ad3d382f371604829afd5d7faf2067d3efbec18a0510c9811"
            ),
        },
    },
}


def validate_kernel_sources() -> None:
    """Require the source generation covered by recorded arithmetic validation."""
    for name, expected in KERNEL_SHA256.items():
        relative = canonical_source_reference("src/gleipnir/" + name)
        source = Path(__file__).resolve().parents[4] / relative
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"validated native FP4 kernel source changed: {name}")


def validate_native_fp4_config(student: Mapping[str, Any]) -> bool:
    """Allow receipt reuse only for the unchanged selected arithmetic/envelope."""
    training = student["training"]
    enabled = training.get("native_fp4_mlp", False)
    if type(enabled) is not bool:
        raise ValueError("native_fp4_mlp must be boolean")
    if not enabled:
        return False
    quantization = student.get("quantization", {})
    adaptive = training.get("adaptive_microbatching", {})
    expected = {
        "sequence_packing": True,
        "native_fp4_mlp_parity_policy": "selected_finite",
        "gated_delta_backend": "flashqla",
        "gated_delta_parity_policy": "selected_finite",
        "gradient_checkpointing": False,
        "selective_torch_compile_policy": "full_attention_and_linear_shell",
        "selective_torch_compile_backend": "inductor",
        "selective_torch_compile_mode": "default",
        "selective_torch_compile_dynamic": True,
        "allow_unspec_int_on_nn_module": False,
        "eager_attention_interface": False,
        "packing_compile_cache_limit": 64,
        "packed_attention_backend": "flash_attention_4",
        "packed_attention_version": "4.0.0b33",
        "packing_learning_gradient_tolerance": 0.05,
        "startup_validation_reference_sha256": REFERENCE_SHA256,
        "per_device_train_batch_size": 32,
        "gradient_accumulation_steps": 1,
    }
    if (
        any(training.get(key) != value for key, value in expected.items())
        or not training.get("startup_validation_reference")
        or training.get("packing_timing_authority") is not None
        or quantization.get("full_bf16_lora") is not True
        or quantization.get("enabled", True)
        or quantization.get("mlp_precision") != "bf16"
        or quantization.get("fp4_mlp_lora", False)
        or student.get("model") != "Qwen/Qwen3.5-4B"
        or student.get("model_revision") != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        or student.get("lora", {}).get("r") != 128
        or student.get("lora", {}).get("alpha") != 256
        or not 0 < int(student.get("max_length", 0)) <= 29696
        or adaptive.get("enabled") is not True
        or adaptive.get("max_padded_tokens") != 16384
        or adaptive.get("max_micro_batch_size") != 8
        or student.get("lora", {}).get("dropout") != 0
    ):
        raise ValueError("native FP4 requires the checksum-bound selected B200 recipe")
    return True


def native_fp4_environment(environment: dict[str, str], root: Path) -> dict[str, str]:
    """Expose the retained cuDNN overlay and reuse the network-volume caches."""
    result = dict(environment)
    overlay = root / ".cache/kernels/nvidia_mxfp8"
    frontend = str(overlay / "frontend")
    paths = [p for p in result.get("PYTHONPATH", "").split(":") if p and p != frontend]
    result["PYTHONPATH"] = ":".join([*paths, frontend])
    result["LD_LIBRARY_PATH"] = ":".join(
        [str(overlay / "runtime/nvidia/cudnn/lib"), result.get("LD_LIBRARY_PATH", "")]
    ).rstrip(":")
    shared = root / ".cache/training/shared"
    for variable, directory in (
        ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
        ("TRITON_CACHE_DIR", "triton"),
        ("TILELANG_CACHE_DIR", "tilelang"),
        ("TVM_CACHE_DIR", "tvm"),
    ):
        result.setdefault(variable, str(shared / "gpu-0" / directory))
    result.update(
        CUDNN_FRONTEND_ENABLE_FROST_ENGINES="1",
        CUDNN_FRONTEND_COMPILED_CACHE=str(shared / "cudnn_frontend"),
        CUDNN_FRONTEND_COMPILED_CACHE_MAX_BYTES="0",
        CUTE_DSL_CACHE_DIR=str(shared / "cute_dsl"),
        GLEIPNIR_NVIDIA_SOURCE=str(overlay / "source"),
    )
    result.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "16")
    result.setdefault("MAX_JOBS", "16")
    result.setdefault("OMP_NUM_THREADS", "4")
    return result


def install_graph_step_boundary(model: Any) -> None:
    """Keep compiled decoder segments within one physical training step."""
    import torch

    if hasattr(model, "_gleipnir_graph_step_boundary"):
        raise ValueError("model graph step boundary is already installed")

    def begin_step(module: Any, args: Any) -> None:
        torch.compiler.cudagraph_mark_step_begin()

    model._gleipnir_graph_step_boundary = model.register_forward_pre_hook(begin_step)


def install_native_fp4_recipe(model: Any) -> dict[str, Any]:
    """Install the warmed hardware packing/fused descale recipe on all 32 MLPs."""
    from gleipnir.cudnn_fp4_mlp import install_fp4_mlp

    modules = [m for m in model.modules() if type(m).__name__ == "Qwen3_5MLP"]
    if len(modules) != 32:
        raise ValueError("native FP4 recipe requires exactly 32 Qwen MLPs")
    installation = install_fp4_mlp(model, hardware_packing=True, fused_descale=True)
    install_graph_step_boundary(model)
    installation["cudagraph_step_boundary"] = "physical_model_forward"
    model._gleipnir_fp4_mlp_installation = installation
    return installation


def accept_selected_canary(receipt: dict[str, Any]) -> None:
    """Check historical finite/isolation acceptance without relabeling parity."""
    from gleipnir.packed_training_screen import _accept_canary

    checked = _accept_canary(deepcopy(receipt), 0.05, "selected native FP4 recipe")
    if not checked["accepted_for_timing_comparison"]:
        raise ValueError("native FP4 receipt lacks finite isolated gradients")


def validate_native_fp4_reference(metadata: dict[str, Any], digest: str) -> None:
    """Bind the historical finite timing receipt and exact kernel arithmetic."""
    from gleipnir.packed_benchmark import summarize

    if digest != REFERENCE_SHA256:
        raise ValueError("native FP4 requires its validated warmed receipt")
    summarize(metadata, 10, accept_timing=True)
    native = metadata["quantization"]["full_bf16_lora"]["native_fp4_mlp"]
    if (
        not native.get("hardware_packing")
        or not native.get("fused_descale")
        or native.get("cudagraph_step_boundary") != "physical_model_forward"
        or metadata["selective_torch_compile"].get("mode") != "default"
        or metadata["sequence_packing"]["learning_gradient_tolerance"] != 0.05
    ):
        raise ValueError("native FP4 reference arithmetic changed")
    for name in ("eager_canary", "compiled_canary"):
        accept_selected_canary(metadata["sequence_packing"][name])
    validate_kernel_sources()


def verify_native_fp4_runtime() -> dict[str, Any]:
    """Verify the pinned native GEMM runtime; no model numerical probes."""
    from importlib.metadata import version

    import cudnn
    import torch
    from cudnn.gated_attention_block.kernels import proj_gemm

    if (
        cudnn.__version__ != "1.31.0"
        or cudnn.backend_version() != 92600
        or torch.version.cuda != "13.0"
        or version("triton") != "3.7.1"
        or version("peft") != "0.19.1"
        or hashlib.sha256(Path(proj_gemm.__file__).read_bytes()).hexdigest()
        != "7207e3da1894956bbac3cf5c0d491d142b924a1d0c5c6fcce35d3b748d8ba447"
    ):
        raise ValueError("native FP4 runtime changed; use fresh startup validation")
    return {
        "cudnn_frontend": cudnn.__version__,
        "cudnn_backend": cudnn.backend_version(),
        "cuda": torch.version.cuda,
        "triton": version("triton"),
        "peft": version("peft"),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "flash_attn_4": version("flash-attn-4"),
        "cutlass_dsl": version("nvidia-cutlass-dsl"),
        "cudnn_proj_gemm_sha256": (
            hashlib.sha256(Path(proj_gemm.__file__).read_bytes()).hexdigest()
        ),
        "cache_paths": {k: v for k, v in os.environ.items() if "CACHE" in k},
    }
