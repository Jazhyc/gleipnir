"""Training-safe fused Qwen head RMSNorm/RoPE/MXFP8 attention producer."""

from __future__ import annotations

from typing import Any

import torch


def _backward(source, weight, cos, sin, grad, eps):
    import triton

    # Import the persistent JIT definition only in the isolated GPU runtime.
    from gleipnir.nvidia_mxfp8_norm_rope_kernel import norm_rope_backward

    output = torch.empty(source.shape, device=source.device, dtype=source.dtype)
    total, heads, _ = source.shape
    norm_rope_backward[(triton.cdiv(total, 16), heads)](
        source,
        weight,
        cos,
        sin,
        grad,
        output,
        total,
        heads,
        source.stride(0),
        source.stride(1),
        grad.stride(0),
        grad.stride(1),
        eps,
        cos.shape[1],
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


class _NormRopeGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, source, weight, cos, sin, eps):
        # Private carrier: consumed exclusively with the prepared FP8 payloads.
        # No normalized BF16 tensor is materialized or read by native attention.
        ctx.save_for_backward(source, weight, cos, sin)
        ctx.eps = eps
        return source

    @staticmethod
    def backward(ctx: Any, grad):
        source, weight, cos, sin = ctx.saved_tensors
        return (
            _backward(source, weight, cos, sin, grad, ctx.eps),
            None,
            None,
            None,
            None,
        )


class _PreparedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q, k, v, cumulative, maximum, prepared):
        from gleipnir.nvidia_mxfp8_fused_attention import forward_plan
        from gleipnir.nvidia_mxfp8_varlen_repack import singleton_forward

        qs, ks, vs = prepared
        qr, qc, qfa, qfb, qfc, qp, _ = qs
        kr, kc, kfa, kfb, kfc, kp, _ = ks
        vr, vc, vfa, vfb, _, _, vp = vs
        out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        lse = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
        plan = forward_plan(q.shape[1], k.shape[1], q.device)
        workspace = torch.empty(
            max(1, plan.scratch_workspace_bytes()), device=q.device, dtype=torch.uint8
        )
        plan.execute(
            qr,
            kr,
            vc,
            out,
            lse_tensor=lse,
            seq_q_lens=cumulative,
            seq_kv_lens=cumulative,
            sf_q=qp,
            sf_k=kp,
            sf_v=vp,
            workspace=workspace,
        )
        singleton_forward(v, out, cumulative)
        ctx.square = True
        ctx.maximum = maximum
        ctx.scale = 0.0625
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
            out,
            lse,
            cumulative,
        )
        return out

    @staticmethod
    def backward(ctx: Any, grad):
        from gleipnir.nvidia_mxfp8_fused_attention import _FusedMxfp8

        dq, dk, dv, _, _, _, _ = _FusedMxfp8.backward(ctx, grad)
        return dq, dk, dv, None, None, None


def norm_rope_attention(
    q, k, v, q_weight, k_weight, cos, sin, cumulative, maximum, *, eps=1e-6
):
    """Consume raw BF16 projected Q/K, with frozen Qwen norm and partial RoPE."""
    from gleipnir.nvidia_mxfp8_attention import validate_inputs
    from gleipnir.nvidia_mxfp8_fused_attention import validate_layout
    from gleipnir.nvidia_mxfp8_fused_quantize import prepare

    validate_inputs(q, k, v)
    validate_layout(q, cumulative, maximum)
    qs = prepare(
        q,
        cumulative,
        maximum,
        square=True,
        norm_weight=q_weight,
        cos=cos,
        sin=sin,
        eps=eps,
    )
    ks = prepare(
        k,
        cumulative,
        maximum,
        square=True,
        norm_weight=k_weight,
        cos=cos,
        sin=sin,
        eps=eps,
    )
    vs = prepare(v, cumulative, maximum, square=True)
    qc = _NormRopeGradient.apply(q, q_weight, cos, sin, eps)
    kc = _NormRopeGradient.apply(k, k_weight, cos, sin, eps)
    return _PreparedAttention.apply(qc, kc, v, cumulative, maximum, (qs, ks, vs))
