"""Reuse recorded recipe validation without claiming new diagnostic passes."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from gleipnir.flashqla_training import (
    load_flashqla,
    make_flashqla_kernel,
    make_precision_boundary,
)


def validation_reference(
    path: Path,
    *,
    packed_attention_backend: str | None = None,
    packed_attention_version: str | None = None,
    learning_gradient_tolerance: float | None = None,
    expected_sha256: str | None = None,
    verify_runtime: bool = False,
    native_fp4_mlp: bool = False,
) -> dict[str, Any]:
    """Load a completed packed recipe receipt for explicitly selected reuse."""
    if native_fp4_mlp and packed_attention_backend == "sdpa":
        return composite_native_fp4_sdpa_reference(
            path, expected_sha256=expected_sha256, verify_runtime=verify_runtime
        )
    contents = path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("validation reference checksum drift")
    metadata = json.loads(contents)
    native = (
        metadata.get("quantization", {}).get("full_bf16_lora", {}).get("native_fp4_mlp")
    )
    if bool(native) != native_fp4_mlp:
        raise ValueError("validation reference MLP precision drift")
    packing = metadata["sequence_packing"]
    backend = metadata["gated_delta_backend"]
    checkpointing = metadata["gradient_checkpointing"]
    attention = packing.get("attention_backend", "sdpa")
    version = packing.get("attention_version")
    tolerance = packing.get("learning_gradient_tolerance")
    if packed_attention_backend is not None and (
        (attention, version, tolerance)
        != (
            packed_attention_backend,
            packed_attention_version,
            learning_gradient_tolerance,
        )
    ):
        raise ValueError("validation reference attention/acceptance policy drift")
    if attention == "flash_attention_4":
        if (
            version != "4.0.0b33"
            or tolerance != (0.05 if native_fp4_mlp else 0.10)
            or metadata["model"] != "Qwen/Qwen3.5-4B"
            or checkpointing
            or not metadata["quantization"].get("full_bf16_lora", {}).get("verified")
        ):
            raise ValueError("reference does not validate selected BF16 FA4 recipe")
        from gleipnir.packed_training_screen import _accept_canary

        if native_fp4_mlp:
            from gleipnir.native_fp4_training import validate_native_fp4_reference

            if expected_sha256 is None:
                raise ValueError("native FP4 receipt must be checksum-bound")
            validate_native_fp4_reference(metadata, digest)
        else:
            for name in ("eager_canary", "compiled_canary"):
                _accept_canary(deepcopy(packing[name]), tolerance)
        packing_passed = packing["preflight"]["passed"]
        if verify_runtime:
            import torch

            expected_packages = {
                "torch": "2.11.0",
                "transformers": "5.14.1",
                "flash-attn-4": "4.0.0b33",
                "nvidia-cutlass-dsl": "4.8.0",
                "nvidia-cutlass-dsl-libs-cu13": "4.8.0",
                "apache-tvm-ffi": "0.1.11",
            }
            if (
                any(
                    package_version(name).split("+")[0] != expected
                    for name, expected in expected_packages.items()
                )
                or torch.cuda.get_device_name() != "NVIDIA B200"
            ):
                raise ValueError(
                    "FA4 validated hardware/software changed; "
                    "use fresh startup validation"
                )
    elif attention == "sdpa" and version is None and tolerance is None:
        packing_passed = all(
            packing[k]["passed"]
            for k in ("eager_canary", "compiled_canary", "preflight")
        )
    else:
        raise ValueError("unsupported validation reference attention policy")
    if checkpointing and (
        metadata["model"] != "Qwen/Qwen3.5-9B"
        or metadata.get("gradient_checkpointing_policy") != "all"
        or metadata.get("checkpointed_layer_indices") != list(range(32))
        or metadata.get("selective_torch_compile", {}).get("policy")
        != "checkpointed_full_attention_and_linear_shell"
    ):
        raise ValueError("reference does not validate the checkpointed 9B recipe")
    if (
        metadata["training_state"]["global_step"] <= 0
        or metadata["quantization"]["enabled"]
        or backend["backend"] != "flashqla"
        or backend["replaced_layers"] != 24
        or backend["boundary_policy"] != "bf16_fp32_gates_norm"
        or backend["auto_cp"]
        or not backend["finite"]
        or not packing_passed
        or packing["max_packed_tokens"] != 16384
    ):
        raise ValueError("reference does not validate the selected packed BF16 recipe")
    native_runtime = {}
    if native_fp4_mlp and verify_runtime:
        from gleipnir.native_fp4_training import verify_native_fp4_runtime

        native_runtime = verify_native_fp4_runtime()
    native_sources = {}
    if native_fp4_mlp:
        from gleipnir.native_fp4_training import RUNTIME_SHAPE_VALIDATION

        native_sources = deepcopy(RUNTIME_SHAPE_VALIDATION)
    return {
        "performed_this_run": False,
        "policy": (
            "reuse_selected_finite_native_fp4_recipe"
            if native_fp4_mlp
            else "reuse_validated_recipe_at_user_request"
        ),
        **(
            {
                "native_fp4_mlp": native,
                "native_fp4_mlp_parity_policy": "selected_finite",
                "waived_checks": ["loss_parity", "gradient_relative_l2"],
                "native_fp4_runtime": native_runtime,
                "native_fp4_source_validation": native_sources,
            }
            if native_fp4_mlp
            else {}
        ),
        "reference_path": str(path),
        "reference_sha256": digest,
        "model": metadata["model"],
        "model_revision": metadata["model_revision"],
        "gradient_checkpointing": checkpointing,
        "attention_backend": attention,
        "attention_version": version,
        "learning_gradient_tolerance": tolerance,
        "runtime_verified_this_run": verify_runtime
        and attention == "flash_attention_4",
        "reference_packing": {
            k: packing[k] for k in ("eager_canary", "compiled_canary", "preflight")
        },
        "reference_backend": backend,
        "reference_adaptive_canary": metadata.get("adaptive_microbatching", {}).get(
            "gradient_canary"
        ),
        "reference_compile_canary": metadata.get("selective_torch_compile", {}).get(
            "canary"
        ),
    }


def composite_native_fp4_sdpa_reference(
    path: Path, *, expected_sha256: str | None, verify_runtime: bool
) -> dict[str, Any]:
    """Native FP4 MLP with SDPA packing: two receipts, each within its own scope.

    The accepted BF16/SDPA receipt covers attention packing, FlashQLA and the
    compile/adaptive canaries. The checksum-bound FP4 receipt covers the MLP
    arithmetic under the user's finite acceptance (loss and gradient parity
    waived, isolation enforced). No fresh probe of the combination is claimed.
    """
    from gleipnir.training.backends.native_fp4 import (
        REFERENCE,
        REFERENCE_SHA256,
        RUNTIME_SHAPE_VALIDATION,
        validate_native_fp4_reference,
        verify_native_fp4_runtime,
    )
    from gleipnir.training.backends.native_fp4_sdpa import SDPA_REFERENCE_SHA256

    if expected_sha256 != SDPA_REFERENCE_SHA256:
        raise ValueError("native FP4 SDPA composite requires the accepted SDPA receipt")
    attention = validation_reference(
        path,
        packed_attention_backend="sdpa",
        expected_sha256=SDPA_REFERENCE_SHA256,
    )
    fp4_path = Path(__file__).resolve().parents[3] / REFERENCE
    fp4_contents = fp4_path.read_bytes()
    fp4_metadata = json.loads(fp4_contents)
    validate_native_fp4_reference(
        fp4_metadata, hashlib.sha256(fp4_contents).hexdigest()
    )
    return {
        **attention,
        "policy": "reuse_composite_sdpa_packing_and_native_fp4_mlp",
        "native_fp4_mlp": fp4_metadata["quantization"]["full_bf16_lora"][
            "native_fp4_mlp"
        ],
        "native_fp4_mlp_parity_policy": "selected_finite",
        "waived_checks": ["loss_parity", "gradient_relative_l2"],
        "native_fp4_runtime": verify_native_fp4_runtime() if verify_runtime else {},
        "native_fp4_source_validation": deepcopy(RUNTIME_SHAPE_VALIDATION),
        "native_fp4_reference_path": str(fp4_path),
        "native_fp4_reference_sha256": REFERENCE_SHA256,
        "composite_scope": {
            "attention_packing_flashqla_compile": "bf16_sdpa_receipt",
            "mlp_arithmetic": "native_fp4_receipt",
            "fresh_combined_probe": False,
        },
    }


def reused_diagnostic_view(metadata: dict[str, Any]) -> dict[str, Any]:
    """Expose referenced diagnostic results for validation, preserving raw receipts."""
    reference = metadata.get("startup_validation")
    if not reference:
        return metadata
    result = {**metadata}
    result["sequence_packing"] = {
        **metadata["sequence_packing"],
        **reference["reference_packing"],
    }
    result["gated_delta_backend"] = {
        **metadata["gated_delta_backend"],
        **reference["reference_backend"],
    }
    for section, key, reference_key in (
        ("adaptive_microbatching", "gradient_canary", "reference_adaptive_canary"),
        ("selective_torch_compile", "canary", "reference_compile_canary"),
    ):
        result[section] = {
            **metadata.get(section, {}),
            key: reference.get(reference_key),
        }
    return result


def skipped_diagnostic(reference: dict, name: str) -> dict:
    """A reused result is explicitly distinct from a fresh pass."""
    return {
        "performed_this_run": False,
        "status": "reused_validation",
        "reference_sha256": reference["reference_sha256"],
        "reference_result": reference["reference_packing"][name],
    }


def install_validated_flashqla(model: Any, reference: dict) -> dict:
    """Bind the verified selected kernel, without model forward/backward probes."""
    if "gradient_checkpointing" in reference:
        if (
            bool(getattr(model, "is_gradient_checkpointing", False))
            != reference["gradient_checkpointing"]
        ):
            raise ValueError("model checkpointing differs from validated recipe")
        if reference["gradient_checkpointing"]:
            checkpointed = [
                m
                for m in model.modules()
                if getattr(m, "gradient_checkpointing", False)
            ]
            if not checkpointed or any(
                getattr(
                    getattr(m, "_gradient_checkpointing_func", None), "keywords", {}
                ).get("use_reentrant", True)
                for m in checkpointed
            ):
                raise ValueError(
                    "validated 9B recipe requires nonreentrant checkpointing"
                )
    function, receipt = load_flashqla()
    if receipt["revision"] != reference["reference_backend"]["revision"]:
        raise ValueError("kernel revision differs from the validated recipe")
    kernel = make_precision_boundary(
        make_flashqla_kernel(function, auto_cp=False), policy="bf16_fp32_gates_norm"
    )
    modules = [m for m in model.modules() if hasattr(m, "chunk_gated_delta_rule")]
    if len(modules) != 24:
        raise ValueError("expected 24 GDN layers")
    for module in modules:
        module.chunk_gated_delta_rule = kernel
    return {
        **receipt,
        "backend": "flashqla",
        "replaced_layers": 24,
        "boundary_policy": "bf16_fp32_gates_norm",
        "auto_cp": False,
        "parity_policy": "selected_finite",
        "performed_this_run": False,
        "validation_reference": reference["reference_sha256"],
        "reference_result": reference["reference_backend"],
    }
