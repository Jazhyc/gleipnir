"""Experimental MXFP8 attention with fused operand/scale preparation."""

from typing import Any

import torch

from gleipnir.nvidia_mxfp8_attention import validate_inputs
from gleipnir.nvidia_mxfp8_varlen_attention import (
    MAX_BATCH,
    MAX_LENGTH,
    backward_plan,
    forward_plan,
    packed_mxfp8_interface,
)


def validate_layout(q: torch.Tensor, cumulative: torch.Tensor, maximum: int) -> None:
    """Retain the original envelope and device checks with one bounds producer."""
    if not q.is_cuda or cumulative.device != q.device:
        raise ValueError("packed MXFP8 requires CUDA inputs and same-device cuts")
    if cumulative.dtype != torch.int32 or cumulative.ndim != 1:
        raise ValueError("cumulative lengths must be a one-dimensional int32 tensor")
    if not cumulative.is_contiguous() or not 2 <= cumulative.numel() <= MAX_BATCH + 1:
        raise ValueError("packed MXFP8 requires contiguous cuts for 1–32 sequences")
    if not isinstance(maximum, int) or isinstance(maximum, bool):
        raise ValueError("maximum length must be a host integer")
    if not 1 <= maximum <= MAX_LENGTH or q.shape[0] > MAX_BATCH * MAX_LENGTH:
        raise ValueError("packed MXFP8 exceeds the declared context envelope")
    from gleipnir.nvidia_mxfp8_fused_quantize import check_boundaries

    check_boundaries(cumulative, q.shape[0], maximum)


class _FusedMxfp8(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q, k, v, cumulative, maximum, scale, square):
        from gleipnir.nvidia_mxfp8_fused_quantize import prepare
        from gleipnir.nvidia_mxfp8_varlen_repack import singleton_forward

        validate_inputs(q, k, v)
        validate_layout(q, cumulative, maximum)
        if scale != 0.0625:
            raise ValueError("fused D256 MXFP8 requires softmax scale 1/16")
        qr, qc, qfa, qfb, qfc, qp, _ = prepare(q, cumulative, maximum, square=square)
        kr, kc, kfa, kfb, kfc, kp, _ = prepare(k, cumulative, maximum, square=square)
        vr, vc, vfa, vfb, _, _, vp = prepare(v, cumulative, maximum, square=square)
        total, hq, _ = q.shape
        output = torch.empty_like(q)
        lse = torch.empty((total, hq), device=q.device, dtype=torch.float32)
        plan = forward_plan(hq, k.shape[1], q.device)
        workspace = torch.empty(
            max(1, plan.scratch_workspace_bytes()), device=q.device, dtype=torch.uint8
        )
        plan.execute(
            qr,
            kr,
            vc,
            output,
            lse_tensor=lse,
            seq_q_lens=cumulative,
            seq_kv_lens=cumulative,
            sf_q=qp,
            sf_k=kp,
            sf_v=vp,
            workspace=workspace,
        )
        singleton_forward(v, output, cumulative)
        ctx.square = square
        ctx.maximum = maximum
        ctx.scale = scale
        ctx.save_for_backward(
            qr,
            qc,
            kr,
            kc,
            vr,
            qfa,
            qfb,
            qfc,
            kfa,
            kfb,
            kfc,
            vfa,
            vfb,
            output,
            lse,
            cumulative,
        )
        return output

    @staticmethod
    def backward(ctx: Any, grad):
        from gleipnir.nvidia_mxfp8_fused_quantize import prepare
        from gleipnir.nvidia_mxfp8_varlen_repack import singleton_backward

        qr, qc, kr, kc, vr, qfa, qfb, qfc, kfa, kfb, kfc, vfa, vfb, out, lse, cu = (
            ctx.saved_tensors
        )
        grad = grad.contiguous()
        dr, dc, dfa, dfb, dfc, _, _ = prepare(grad, cu, ctx.maximum, square=ctx.square)
        scales = (qfa, kfb, kfc, vfb, dfa, qfb, qfc, kfa, vfa, dfb, dfc)
        total, hq, _ = qr.shape
        hk, batch = kr.shape[1], cu.numel() - 1
        dq, dk, dv = (
            torch.empty_like(out),
            torch.empty_like(grad[:, :hk]),
            torch.empty_like(grad[:, :hk]),
        )
        stats = lse.transpose(0, 1).contiguous()
        workspace = torch.zeros(
            batch * hq * ((ctx.maximum + 7) // 8 * 8) * 2 * 4,
            device=grad.device,
            dtype=torch.uint8,
        )
        _, function = backward_plan(hq, hk)
        tensors = (
            qr,
            kr,
            vr,
            out,
            dr,
            stats,
            dq,
            dk,
            dv,
            qc,
            kc,
            dc,
            grad,
            *scales,
            cu,
            workspace,
        )
        function(
            tuple(t.data_ptr() for t in tensors),
            total,
            ctx.maximum,
            batch,
            ctx.scale,
            torch.cuda.current_stream().cuda_stream,
        )
        singleton_backward(grad, dq, dk, dv, cu)
        return dq, dk, dv, None, None, None, None


def packed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative: torch.Tensor,
    maximum: int,
    *,
    scale: float = 0.0625,
    square: bool = False,
) -> torch.Tensor:
    """Execute causal D256 GQA with fused dual or square-block preparation."""
    return _FusedMxfp8.apply(q, k, v, cumulative, maximum, scale, square)


def fused_interface(original, *, square: bool = False):
    """Keep precision modes explicit and retain the existing packed contract."""

    def kernel(*args, **kwargs):
        return packed_attention(*args, **kwargs, square=square)

    return packed_mxfp8_interface(original, kernel)
