"""Opt-in BF16 packing comparison with strict cross-example isolation gates."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from gleipnir.packed_sequences import (
    collate_packed_monitoring,
    forward_packed_monitoring_logits,
    installed_segmented_sdpa,
    packed_partition,
)
from gleipnir.precision_training_screen import run_precision_training_screen
from gleipnir.training_execution_audit import tensor_digest, use_forwards


class PackingCanaryError(ValueError):
    """Retain successful isolation measurements when a later parity gate fails."""

    def __init__(self, message: str, receipt: dict) -> None:
        super().__init__(message)
        self.receipt = receipt


def _layer_comparison(reference: dict, candidate: dict) -> list[dict]:
    results = []
    for name, tensors in reference.items():
        expected = torch.cat(tensors, dim=1).float()
        actual = candidate[name][0].float()
        if actual.shape != expected.shape:
            raise ValueError(f"diagnostic layer shape mismatch: {name}")
        difference = actual - expected
        results.append(
            {
                "module": name,
                "max_absolute": float(difference.abs().max()),
                "relative_l2": float(
                    difference.norm() / expected.norm().clamp_min(1e-30)
                ),
            }
        )
    return results


def validate_learning_tolerance(value: float | None) -> None:
    """Limit the explicit learning diagnostic to the authorized 15% ceiling."""
    if value is not None and (
        type(value) not in (int, float) or not 0.05 <= value <= 0.15
    ):
        raise ValueError(
            "packing learning gradient tolerance must be between .05 and .15"
        )


def validate_compile_cache_limit(value: int | None) -> None:
    """Keep additional compiler variants bounded for the packing diagnostic."""
    if value is not None and (type(value) is not int or not 8 <= value <= 128):
        raise ValueError("packing compile cache limit must be an integer from 8 to 128")


def _accept_canary(receipt: dict, learning_tolerance: float | None) -> dict:
    validate_learning_tolerance(learning_tolerance)
    reference = receipt["independent_loss"]
    candidate = receipt["packed_loss"]
    relative = receipt["adapter_gradient_relative_l2"]
    isolation = bool(receipt["cases"]) and all(
        all(
            math.isfinite(v) and 0 <= v <= limit
            for v, limit in [
                (row["repeat_max_abs"], 1e-6),
                (row["perturb_max_abs"], 1e-6),
                (row["cross_input_grad_max_abs"], 1e-8),
            ]
        )
        and math.isfinite(row["own_input_grad_max_abs"])
        and row["own_input_grad_max_abs"] > 0
        for row in receipt["cases"]
    )
    valid = (
        isolation
        and all(math.isfinite(v) for v in [reference, candidate, relative])
        and relative >= 0
        and abs(candidate - reference) <= 0.02 + 0.02 * abs(reference)
    )
    receipt["passed"] = valid and relative <= 0.05
    receipt["learning_gradient_tolerance"] = learning_tolerance
    receipt["accepted_for_learning_comparison"] = bool(
        valid and learning_tolerance is not None and relative <= learning_tolerance
    )
    if not (receipt["passed"] or receipt["accepted_for_learning_comparison"]):
        raise PackingCanaryError(
            f"packing numerical parity failed: losses {reference}/{candidate}, "
            f"gradient rel L2={relative}",
            receipt,
        )
    return receipt


def packing_isolation_canary(
    model: torch.nn.Module,
    features: list[dict[str, Any]],
    collator: Callable,
    loss_forward: Callable,
    *,
    capture_layers: bool = True,
    learning_tolerance: float | None = None,
) -> dict:
    """Test identical-shape independence and matched singleton adapter gradients.

    Tolerances are fixed before GPU execution. Identical-shape perturbations
    permit 1e-6 absolute logit drift and 1e-8 cross-input gradients; numerical
    singleton/packing parity permits 2% loss drift and 0.05 gradient relative L2.
    """
    validate_learning_tolerance(learning_tolerance)
    device = next(model.parameters()).device
    parameters = [p for p in model.parameters() if p.requires_grad]
    rows = []
    was_training = model.training
    try:
        for lengths in [(1, 3), (63, 65), (127, 129)]:
            items = [dict(features[i]) for i in range(2)]
            for item, length in zip(items, lengths, strict=True):
                item["direct_input_ids"] = item["direct_input_ids"][-length:]
            lengths = tuple(len(f["direct_input_ids"]) for f in items)
            ids = collate_packed_monitoring(items)["direct_input_ids"].to(device)
            altered = ids.clone()
            altered[:, : lengths[0]] = (
                altered[:, : lengths[0]] + 17
            ) % model.config.vocab_size
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                before, _ = forward_packed_monitoring_logits(model, ids, lengths)
                repeat, _ = forward_packed_monitoring_logits(model, ids, lengths)
                after, _ = forward_packed_monitoring_logits(model, altered, lengths)
            repeat_diff = float((before[1].float() - repeat[1].float()).abs().max())
            perturb_diff = float((before[1].float() - after[1].float()).abs().max())
            if (
                not torch.isfinite(before).all()
                or max(repeat_diff, perturb_diff) > 1e-6
            ):
                raise ValueError(f"packed decision leakage: {lengths}, {perturb_diff}")
            # Training mode enables the configured checkpoint recomputation.
            model.train()
            model.zero_grad(set_to_none=True)
            embeds = model.get_input_embeddings()(ids).detach().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = forward_packed_monitoring_logits(
                    model, ids, lengths, inputs_embeds=embeds
                )
                logits[1].float().square().mean().backward()
            if embeds.grad is None or not torch.isfinite(embeds.grad).all():
                raise ValueError("missing/nonfinite packed input gradients")
            cross = float(embeds.grad[:, : lengths[0]].float().abs().max())
            own = float(embeds.grad[:, lengths[0] :].float().abs().max())
            if cross > 1e-8 or own <= 0:
                raise ValueError(f"packed input gradient leakage: {cross}, own={own}")
            rows.append(
                {
                    "lengths": lengths,
                    "repeat_max_abs": repeat_diff,
                    "perturb_max_abs": perturb_diff,
                    "cross_input_grad_max_abs": cross,
                    "own_input_grad_max_abs": own,
                }
            )
        # Equal example weighting and the actual monitoring objective, same backend.
        model.eval()
        model.zero_grad(set_to_none=True)
        activations: dict = {}
        independent_activations: dict = {}
        packed_activations: dict = {}
        hooks = []
        for name, module in model.named_modules() if capture_layers else []:
            parts = name.split(".")
            if (
                (len(parts) >= 2 and parts[-2] == "layers" and parts[-1].isdigit())
                or name.endswith(".layers.0.linear_attn.in_proj_qkv")
                or name.endswith(".layers.0.input_layernorm")
                or name.endswith(".layers.0.linear_attn")
                or name.endswith(".model.norm")
                or name.endswith(".lm_head")
            ):

                def capture(module, args, output, name=name):
                    tensor = output[0] if isinstance(output, tuple) else output
                    activations.setdefault(name, []).append(tensor.detach().cpu())

                hooks.append(module.register_forward_hook(capture))
        independent_loss = 0.0
        try:
            activations = independent_activations
            for item in items:
                loss = loss_forward(collator([item])) / len(items)
                independent_loss += float(loss.detach())
                loss.backward()
            independent = [p.grad.detach().float().clone() for p in parameters]
            model.zero_grad(set_to_none=True)
            activations = packed_activations
            packed_loss = loss_forward(collate_packed_monitoring(items))
            packed_loss.backward()
        finally:
            for hook in hooks:
                hook.remove()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError("missing/nonfinite packed adapter gradients")
        numerator = sum(
            (p.grad.float() - g).square().sum()
            for p, g in zip(parameters, independent, strict=True)
        )
        denominator = sum(g.square().sum() for g in independent)
        relative = float(torch.sqrt(numerator / denominator.clamp_min(1e-30)))
        packed_value = float(packed_loss.detach())
        receipt = {
            "passed": False,
            "execution_mode": "eager" if capture_layers else "compiled",
            "cases": rows,
            "independent_loss": independent_loss,
            "packed_loss": packed_value,
            "adapter_gradient_relative_l2": relative,
            "layers": _layer_comparison(independent_activations, packed_activations),
            "layer_diagnostics": "eager_only"
            if capture_layers
            else "omitted_under_compile",
            "tolerances": {
                "logits_absolute": 1e-6,
                "cross_input_gradient": 1e-8,
                "loss_absolute": 0.02,
                "loss_relative": 0.02,
                "gradient_relative_l2": 0.05,
            },
        }
        return _accept_canary(receipt, learning_tolerance)
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def run_packed_training_screen(**kwargs: Any) -> dict:
    """Replay the ten-update BF16 recipe, using packing by default."""
    metadata = kwargs["metadata"]
    packing_only = metadata.get("packing_only", True)
    if type(packing_only) is not bool:
        raise ValueError("packing_only must be a boolean")
    learning_tolerance = metadata.get("packing_learning_gradient_tolerance")
    validate_learning_tolerance(learning_tolerance)
    cache_limit = metadata.get("packing_compile_cache_limit")
    validate_compile_cache_limit(cache_limit)
    compile_options = (
        {"recompile_limit": cache_limit, "fail_on_recompile_limit_hit": True}
        if cache_limit is not None
        else {}
    )
    if not (
        metadata.get("quantization", {}).get("full_bf16_lora", {}).get("verified")
        and kwargs.get("gated_delta_backend") == "flashqla"
        and not kwargs.get("flashqla_auto_cp")
        and kwargs["steps"] == 10
        and kwargs.get("ten_step_learning_comparison")
    ):
        raise ValueError(
            "packing screen requires the verified uniform BF16 FlashQLA ten-step recipe"
        )
    model = kwargs["model"]
    output = Path(kwargs["output"])
    output.mkdir(parents=True, exist_ok=True)
    original_forwards = [(m, m.forward) for m in model.modules()]
    kernels = [
        (m, m.chunk_gated_delta_rule)
        for m in model.modules()
        if hasattr(m, "chunk_gated_delta_rule")
    ]
    convolutions = [m.causal_conv1d_fn for m, _ in kernels]
    if not convolutions or any(
        fn is None or "causal_conv1d" not in fn.__module__ for fn in convolutions
    ):
        raise ValueError(
            "packed training requires the native boundary-aware convolution"
        )
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    initial = [p.detach().cpu().clone() for _, p in named]
    report = {
        "status": "running",
        "packing_only": packing_only,
        "learning_gradient_tolerance": learning_tolerance,
        "compile_cache_intervention": compile_options,
        "initial_master_sha256": tensor_digest(initial),
        "convolution_kernels": sorted({fn.__module__ for fn in convolutions}),
        "conditions": {},
    }

    def publish():
        (output / "packing_comparison.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    def reset():
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for (_, p), value in zip(named, initial, strict=True):
                p.copy_(value)
        for module, kernel in kernels:
            module.chunk_gated_delta_rule = kernel

    try:
        publish()
        with installed_segmented_sdpa(), torch._dynamo.config.patch(compile_options):
            # Fail before either optimizer trajectory if the installed kernels leak.
            # The existing runner installs the selected FlashQLA precision boundary
            # before invoking this gate, and invokes it again after compilation.
            for condition in ["packed"] if packing_only else ["padded", "packed"]:
                reset()
                if report["conditions"]:
                    # Release the first trajectory's AdamW states before preflight.
                    kwargs["optimizer_factory"]()
                options = dict(kwargs, output=output / condition)
                options["common_probe_collator"] = kwargs["collator"]
                options["packing_canary"] = lambda *, compiled: (
                    packing_isolation_canary(
                        model,
                        sorted(
                            kwargs["features"],
                            key=lambda f: -len(f["direct_input_ids"]),
                        ),
                        kwargs["collator"],
                        kwargs["loss_forward"],
                        capture_layers=not compiled,
                        learning_tolerance=learning_tolerance,
                    )
                )
                if condition == "packed":
                    options["collator"] = collate_packed_monitoring
                    options["partition_strategy"] = lambda lengths: packed_partition(
                        lengths, kwargs["policy"].max_padded_tokens
                    )
                with use_forwards(original_forwards):
                    report["conditions"][condition] = run_precision_training_screen(
                        **options
                    )
                publish()
        if "padded" in report["conditions"]:
            padded = report["conditions"]["padded"]["timing_summary"]
            packed = report["conditions"]["packed"]["timing_summary"]
            report["step_time_reduction_fraction"] = (
                1 - packed["mean_step_seconds"] / padded["mean_step_seconds"]
            )
        report["status"] = "complete"
        publish()
        return report
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        if isinstance(error, PackingCanaryError):
            report["failed_canary"] = error.receipt
        publish()
        raise
    finally:
        reset()
