"""Opt-in, pinned FlashQLA training diagnostics for the native FP4 screen."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

FLASHQLA_REVISION = "da06429d54b0f577de0a638f451ac8f0b395e0ac"
FLASHQLA_TARGET = Path(".cache/kernels/flashqla-da06429")
BOUNDARY_POLICIES = {"bf16", "bf16_fp32_gates_norm", "fp16_fp32_gates_norm"}


def make_flashqla_kernel(function: Callable, *, auto_cp: bool) -> Callable:
    """Adapt only the supported Qwen training API; never swallow extra options."""

    def chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
        state_v_first=False,
    ):
        return function(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            state_v_first=state_v_first,
            auto_cp=auto_cp,
            enable_fwd_cp_cache=True,
        )

    return chunk_gated_delta_rule


def make_bf16_boundary(function: Callable) -> Callable:
    """Cast GDN operands explicitly and restore output dtype; retain FP32 gates."""
    return make_precision_boundary(function, policy="bf16")


def _normalize_qk_fp32(x):
    from fla.modules.l2norm import l2norm

    return l2norm(x.float())


def make_precision_boundary(function: Callable, *, policy: str) -> Callable:
    """Preserve optional FP32 gates/normalization around a half-precision kernel."""
    import torch

    if policy not in BOUNDARY_POLICIES:
        raise ValueError("unknown GDN boundary policy")
    input_dtypes = []
    preserve_fp32 = policy != "bf16"
    dtype = torch.float16 if policy.startswith("fp16") else torch.bfloat16

    def kernel(
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
        state_v_first=False,
    ):
        signature = [str(x.dtype) for x in (q, k, v, g, beta)]
        if signature not in input_dtypes:
            input_dtypes.append(signature)
        output_dtype = q.dtype
        if preserve_fp32 and use_qk_l2norm_in_kernel:
            q, k = _normalize_qk_fp32(q), _normalize_qk_fp32(k)
            use_qk_l2norm_in_kernel = False
        output, state = function(
            q.to(dtype),
            k.to(dtype),
            v.to(dtype),
            g.float(),
            beta.float() if preserve_fp32 else beta.to(dtype),
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            state_v_first=state_v_first,
        )
        return output.to(output_dtype), state

    kernel.input_dtypes = input_dtypes
    return kernel


def load_flashqla() -> tuple[Callable, dict[str, Any]]:
    """Verify isolated source identity and versions before selecting the backend."""
    import flash_qla
    import torch

    target = FLASHQLA_TARGET.resolve()
    if not Path(flash_qla.__file__).resolve().is_relative_to(target):
        raise ValueError("FlashQLA imported outside the isolated target")
    manifest = json.loads((target / "install_manifest.json").read_text())
    if manifest["revision"] != FLASHQLA_REVISION:
        raise ValueError("FlashQLA revision mismatch")
    for filename, expected in manifest["package_sha256"].items():
        if hashlib.sha256((target / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"FlashQLA source drift: {filename}")
    compiler_patches = manifest.get("compiler_patches", [])
    for patch in compiler_patches:
        if (
            hashlib.sha256((target / patch["path"]).read_bytes()).hexdigest()
            != patch["after_sha256"]
        ):
            raise ValueError("TileLang compiler patch drift")
        script = Path("experiments/fp4_stability/patch_tilelang_fp16.py")
        if hashlib.sha256(script.read_bytes()).hexdigest() != patch["script_sha256"]:
            raise ValueError("TileLang compiler patch script drift")
    software = {
        name: importlib.metadata.version(name)
        for name in ("flash-qla", "tilelang", "apache-tvm-ffi")
    }
    if software != {
        "flash-qla": "0.1.3+da06429",
        "tilelang": "0.1.12",
        "apache-tvm-ffi": "0.1.11",
    }:
        raise ValueError(f"FlashQLA dependency mismatch: {software}")
    if torch.cuda.get_device_capability() != (10, 0):
        raise ValueError("this FlashQLA experiment is frozen for B200 SM100")
    return flash_qla.chunk_gated_delta_rule, dict(
        revision=FLASHQLA_REVISION,
        software=software,
        archive_sha256=manifest["archive_sha256"],
        package_sha256=manifest["package_sha256"],
        compiler_patches=compiler_patches,
    )


def tensor_comparison(actual, reference) -> dict[str, float | bool]:
    """Measure relative L2 without hiding nonfinite tensors or zero references."""
    import torch

    a = actual.detach().float()
    r = reference.detach().to(actual.device).float()
    finite = bool(torch.isfinite(a).all() & torch.isfinite(r).all())
    if not finite:
        return dict(
            finite=False,
            reference_norm=0.0,
            error_norm=1e30,
            relative_l2=1e30,
            max_absolute_error=1e30,
        )
    ref_norm = float(torch.linalg.vector_norm(r))
    error_norm = float(torch.linalg.vector_norm(a - r))
    return dict(
        finite=finite,
        reference_norm=ref_norm,
        error_norm=error_norm,
        relative_l2=error_norm / max(ref_norm, 1e-12),
        max_absolute_error=float((a - r).abs().max()),
    )


def install_with_model_canary(
    model,
    batches: list,
    loss_forward: Callable,
    *,
    auto_cp: bool,
    backend: str = "flashqla",
    bf16_boundary: bool = False,
    boundary_policy: str = "bf16",
) -> dict[str, Any]:
    """Compare native-model losses and unclipped master gradients before updates."""
    import torch

    if backend == "flashqla":
        function, receipt = load_flashqla()
        kernel = make_flashqla_kernel(function, auto_cp=auto_cp)
    elif backend == "fla_bf16" and bf16_boundary and not auto_cp:
        kernel, receipt = None, dict(backend="fla_bf16")
    else:
        raise ValueError("unsupported backend/boundary configuration")
    modules = [m for m in model.modules() if hasattr(m, "chunk_gated_delta_rule")]
    originals = [(m, m.chunk_gated_delta_rule) for m in modules]
    if len(modules) != 24 or any(
        not f.__module__.startswith("fla.ops.") for _, f in originals
    ):
        raise ValueError("expected all 24 Qwen4B layers to use pinned FLA")
    if backend == "fla_bf16":
        kernel = originals[0][1]
    if bf16_boundary:
        kernel = make_precision_boundary(kernel, policy=boundary_policy)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    training = model.training
    model.eval()
    reference_losses = []
    with torch.no_grad():
        reference_losses = [float(loss_forward(b).detach()) for b in batches]
    model.zero_grad(set_to_none=True)
    loss_forward(batches[-1]).backward()
    reference_gradients = []
    for name, p in named:
        if p.grad is None or not bool(torch.isfinite(p.grad).all()):
            raise FloatingPointError(f"missing/nonfinite FLA gradient: {name}")
        reference_gradients.append(p.grad.detach().cpu().clone())
    model.zero_grad(set_to_none=True)
    passed = False
    try:
        for module in modules:
            module.chunk_gated_delta_rule = kernel
        with torch.no_grad():
            candidate_losses = [float(loss_forward(b).detach()) for b in batches]
        loss_forward(batches[-1]).backward()
        gradients = {}
        for (name, p), reference in zip(named, reference_gradients, strict=True):
            if p.grad is None:
                raise FloatingPointError(f"missing FlashQLA gradient: {name}")
            gradients[name] = tensor_comparison(p.grad, reference)
        relative_l2 = (
            sum(x["error_norm"] ** 2 for x in gradients.values())
            / max(sum(x["reference_norm"] ** 2 for x in gradients.values()), 1e-24)
        ) ** 0.5
        loss_passed = all(
            abs(r - a) <= 0.01 + 0.01 * abs(r)
            for r, a in zip(reference_losses, candidate_losses, strict=True)
        )
        passed = (
            loss_passed
            and all(x["finite"] for x in gradients.values())
            and relative_l2 <= 0.05
        )
        receipt.update(
            backend=backend,
            bf16_boundary=bf16_boundary,
            boundary_policy=boundary_policy,
            original_input_dtypes=getattr(kernel, "input_dtypes", None),
            auto_cp=auto_cp,
            replaced_layers=len(modules),
            reference_losses=reference_losses,
            candidate_losses=candidate_losses,
            gradient_relative_l2=relative_l2,
            gradient_gate=0.05,
            gradients=gradients,
            passed=passed,
        )
        if not passed and backend == "flashqla" and bf16_boundary:
            # Keep the original pass/fail result. These bounded failure probes
            # never update adapters and cannot authorize training past the gate.
            try:
                cast_fla = make_precision_boundary(
                    originals[0][1], policy=boundary_policy
                )
                model.zero_grad(set_to_none=True)
                for module in modules:
                    module.chunk_gated_delta_rule = cast_fla
                with torch.no_grad():
                    cast_losses = [float(loss_forward(b).detach()) for b in batches]
                loss_forward(batches[-1]).backward()
                cast_gradients = {
                    name: tensor_comparison(p.grad, reference)
                    for (name, p), reference in zip(
                        named, reference_gradients, strict=True
                    )
                    if p.grad is not None
                }
                cast_l2 = (
                    sum(x["error_norm"] ** 2 for x in cast_gradients.values())
                    / max(
                        sum(x["reference_norm"] ** 2 for x in cast_gradients.values()),
                        1e-24,
                    )
                ) ** 0.5
                receipt["fla_bf16_failure_control"] = dict(
                    losses=cast_losses,
                    gradient_relative_l2=cast_l2,
                    gradients=cast_gradients,
                    all_gradients_present=len(cast_gradients) == len(named),
                )
                model.zero_grad(set_to_none=True)
                shadows = []
                for index, (module, original) in enumerate(originals):

                    def shadow(*args, original=original, index=index, **kwargs):
                        reference_output, reference_state = original(*args, **kwargs)
                        candidate_output, _ = kernel(*args, **kwargs)
                        cast_output, _ = cast_fla(*args, **kwargs)
                        shadows.append(
                            dict(
                                linear_layer=index,
                                output_shape=list(reference_output.shape),
                                flashqla=tensor_comparison(
                                    candidate_output, reference_output
                                ),
                                fla_bf16=tensor_comparison(
                                    cast_output, reference_output
                                ),
                            )
                        )
                        return reference_output, reference_state

                    module.chunk_gated_delta_rule = shadow
                # Returning original FLA outputs prevents candidate error from
                # changing later-layer inputs during these shadow comparisons.
                with torch.no_grad():
                    for i in sorted({0, len(batches) // 2, len(batches) - 1}):
                        loss_forward(batches[i])
                receipt["shadow_outputs_on_original_path"] = shadows
            except Exception as error:
                receipt["failure_diagnostic_error"] = f"{type(error).__name__}: {error}"
        return receipt
    finally:
        model.zero_grad(set_to_none=True)
        model.train(training)
        if not passed:
            for module, original in originals:
                module.chunk_gated_delta_rule = original
