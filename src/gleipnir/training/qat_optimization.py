"""Scoped systems optimizations for the frozen merged-weight QAD reference."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

from gleipnir.training import quantization_recovery as reference


def native_fp8_linear(codes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Accumulate E4M3 products in FP32; apply PTPC descales separately."""
    one = torch.ones((), device=codes.device, dtype=torch.float32)
    with torch.autocast(codes.device.type, enabled=False):
        return torch._scaled_mm(
            codes,
            weights.t(),
            scale_a=one,
            scale_b=one,
            out_dtype=torch.float32,
            use_fast_accum=False,
        )


def fp8_forward(group: Any, x: torch.Tensor, index: int) -> torch.Tensor:
    group.prepare()
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    start, end = sum(group.widths[:index]), sum(group.widths[: index + 1])
    with torch.no_grad():
        xq, scale = reference.fp8_decode(flat, weight=False)
        codes = (flat.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        output = (
            native_fp8_linear(codes, group.codes[start:end])
            * scale
            * group.scales[start:end].t()
        ).to(torch.bfloat16)
    result = reference.low_rank_ste(
        flat, group.modules[index], output, xq, group.decoded[start:end]
    )
    return result.reshape(*x.shape[:-1], end - start)


class LeanPackedMLP(reference._PackedMLP):
    """Same forward and STE; omit the unused down-output recomputation."""

    @staticmethod
    def backward(ctx, grad):
        from gleipnir.kernels.fp4.cudnn_fp4_gemm import PackedNvfp4

        ac, asc, ai, hc, hsc, hi, *masters = ctx.saved_tensors
        a, h = PackedNvfp4(ac, asc, ai), PackedNvfp4(hc, hsc, hi)
        pair, down = ctx.pair, ctx.down
        with torch.no_grad():
            xq, hq = reference.decode_fp4(a), reference.decode_fp4(h)
            gateup = reference.native_linear(a, pair.packed)
        with torch.enable_grad():
            x = xq.detach().requires_grad_(True)
            operands = [p.detach().requires_grad_(True) for p in masters]
            ga, gb, ua, ub, da, db = operands
            width = pair.widths[0]
            gate = reference._LowRankSTE.apply(
                x,
                ga,
                gb,
                gateup[:, :width],
                xq,
                pair.decoded[:width],
                reference.adapter_operands(pair.modules[0])[2],
            )
            up = reference._LowRankSTE.apply(
                x,
                ua,
                ub,
                gateup[:, width:],
                xq,
                pair.decoded[width:],
                reference.adapter_operands(pair.modules[1])[2],
            )
            hidden = F.silu(gate) * up
            # The STE backward depends only on xq/Wq/A/B and incoming grad.
            # Its returned output is never read in this local backward graph.
            output_shape = (len(hq), down.decoded.shape[0])
            placeholder = hq.new_empty(output_shape)
            out = reference._LowRankSTE.apply(
                hidden,
                da,
                db,
                placeholder,
                hq,
                down.decoded,
                reference.adapter_operands(down.modules[0])[2],
            )
            gradients = torch.autograd.grad(out, (x, *operands), grad.reshape_as(out))
        return gradients[0].reshape(ctx.input_shape), *gradients[1:], None, None, None


def shared_fp4_forward(group: Any, x: torch.Tensor, index: int) -> torch.Tensor:
    """Share frozen GDN QKV/Z forward; retain separate STE backward calls."""
    group.prepare()
    if group.operator_check is None:
        return reference.ProjectionGroup.forward(group, x, index)
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    pending = getattr(group, "_shared_projection", None)
    if (
        index == 1
        and pending is not None
        and pending[0] is x
        and pending[1] == x._version
        and pending[2] == group.key
    ):
        output = pending[3]
    else:
        with torch.no_grad():
            output = reference.native_linear(
                reference.pack_activation(flat), group.packed
            )
    group._shared_projection = (
        (x, x._version, group.key, output) if index == 0 else None
    )
    start, end = sum(group.widths[:index]), sum(group.widths[: index + 1])
    result = reference.low_rank_ste(
        flat,
        group.modules[index],
        output[:, start:end],
        flat.new_empty((0, flat.shape[-1])),
        group.decoded[start:end],
    )
    return result.reshape(*x.shape[:-1], end - start)


@contextmanager
def optimized_path(
    model: Any,
    *,
    fp8: bool = False,
    mlp: bool = True,
    gdn: bool = True,
) -> Iterator[None]:
    """Restore all methods on exit, retaining model parameters and kernel caches."""
    original_mlp = reference._PackedMLP
    methods = []
    try:
        if mlp:
            reference._PackedMLP = LeanPackedMLP
        for group in model._gleipnir_qad_groups:
            method = None
            if fp8 and group.precision == "fp8":
                method = fp8_forward
            elif (
                gdn
                and group.precision == "fp4"
                and len(group.modules) == 2
                and all(not hasattr(m, "lora_A") for m in group.modules)
            ):
                method = shared_fp4_forward
            if method is not None:
                methods.append((group, group.forward))
                group.forward = MethodType(method, group)
        yield
    finally:
        reference._PackedMLP = original_mlp
        for group, method in methods:
            group.forward = method
            if hasattr(group, "_shared_projection"):
                del group._shared_projection
