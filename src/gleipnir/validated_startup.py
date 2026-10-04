"""Reuse recorded recipe validation without claiming new diagnostic passes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from gleipnir.flashqla_training import (
    load_flashqla,
    make_flashqla_kernel,
    make_precision_boundary,
)


def validation_reference(path: Path) -> dict[str, Any]:
    """Load a completed packed BF16 recipe receipt for explicitly requested reuse."""
    metadata = json.loads(path.read_text())
    packing = metadata["sequence_packing"]
    backend = metadata["gated_delta_backend"]
    checkpointing = metadata["gradient_checkpointing"]
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
        or not all(
            packing[k]["passed"]
            for k in ("eager_canary", "compiled_canary", "preflight")
        )
        or packing["max_packed_tokens"] != 16384
    ):
        raise ValueError("reference does not validate the selected packed BF16 recipe")
    return {
        "performed_this_run": False,
        "policy": "reuse_validated_recipe_at_user_request",
        "reference_path": str(path),
        "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model": metadata["model"],
        "model_revision": metadata["model_revision"],
        "gradient_checkpointing": checkpointing,
        "reference_packing": {
            k: packing[k] for k in ("eager_canary", "compiled_canary", "preflight")
        },
        "reference_backend": backend,
    }


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
