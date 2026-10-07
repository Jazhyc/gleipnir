"""Experimental direct THD MXFP8 causal training with runtime sequence cuts."""

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from gleipnir.nvidia_mxfp8_attention import UPSTREAM_REVISION, _cudnn, validate_inputs

MAX_LENGTH = 131072
MAX_BATCH = 32


def validate_layout(q: torch.Tensor, cumulative: torch.Tensor, maximum: int) -> None:
    """Check host metadata; device offset contents are a separately tested contract."""
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
    lengths = cumulative[1:] - cumulative[:-1]
    valid = (cumulative[0] == 0) & (cumulative[-1] == q.shape[0])
    valid = valid & ((lengths > 0) & (lengths <= maximum)).all()
    torch._assert_async(valid, "invalid packed MXFP8 device boundaries")


@lru_cache(maxsize=8)
def forward_plan(hq: int, hk: int, device: torch.device):
    from cudnn.api_base import TensorDesc
    from cudnn.sdpa.fwd import SdpaFwdDslSm100

    _cudnn()

    def desc(heads, dtype):
        return TensorDesc(
            dtype,
            (MAX_BATCH, heads, MAX_LENGTH, 256),
            (MAX_LENGTH * heads * 256, 256, heads * 256, 1),
            (3, 1, 2, 0),
            device,
        )

    lse = TensorDesc(
        torch.float32,
        (MAX_BATCH, hq, MAX_LENGTH),
        (MAX_LENGTH * hq, 1, hq),
        (1, 2, 0),
        device,
    )
    plan = SdpaFwdDslSm100(
        desc(hq, torch.float8_e4m3fn),
        desc(hk, torch.float8_e4m3fn),
        desc(hk, torch.float8_e4m3fn),
        desc(hq, torch.bfloat16),
        sample_lse=lse,
        is_causal=True,
        scale_softmax=0.0625,
        thd=True,
        seq_q_lens_present=False,
        seq_kv_lens_present=True,
        cu_seq_q_lens=True,
        cu_seq_kv_lens=True,
        cga=1,
        split_kv=1,
        dtype_o=torch.bfloat16,
    )
    plan.check_support()
    plan.compile()
    return plan


@lru_cache(maxsize=8)
def backward_plan(hq: int, hk: int):
    from cudnn.frost.compiled_cache import positional_entry

    from gleipnir.nvidia_mxfp8_varlen_host import compile_packed_backward

    path = Path(__file__).with_name("nvidia_mxfp8_varlen_host.py")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    owner = compile_packed_backward((hq, hk), f"{UPSTREAM_REVISION}:{digest}:{hq}:{hk}")
    fn = positional_entry(owner)
    if fn is None:
        raise RuntimeError("packed backward requires a TVM FFI entry")
    return owner, fn


class _PackedMxfp8(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q, k, v, cumulative, maximum, scale):
        from gleipnir.nvidia_mxfp8_varlen_quantize import quantize
        from gleipnir.nvidia_mxfp8_varlen_repack import singleton_forward

        validate_inputs(q, k, v)
        validate_layout(q, cumulative, maximum)
        if scale != 0.0625:
            raise ValueError("the bounded packed D256 screen fixes softmax scale 1/16")
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        qr, sqr, pqr = quantize(q, cumulative, maximum, False)
        qc, sqc, _ = quantize(q, cumulative, maximum, True)
        kr, skr, pkr = quantize(k, cumulative, maximum, False)
        kc, skc, _ = quantize(k, cumulative, maximum, True)
        vr, svr, _ = quantize(v, cumulative, maximum, False)
        vc, _, pvc = quantize(v, cumulative, maximum, True)
        total, hq, _ = q.shape
        hk = k.shape[1]
        output = torch.empty_like(q)
        lse = torch.empty((total, hq), device=q.device, dtype=torch.float32)
        plan = forward_plan(hq, hk, q.device)
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
            sf_q=pqr,
            sf_k=pkr,
            sf_v=pvc,
            workspace=workspace,
        )
        singleton_forward(v, output, cumulative)
        ctx.maximum = maximum
        ctx.scale = scale
        ctx.save_for_backward(
            qr, qc, kr, kc, vr, sqr, sqc, skr, skc, svr, output, lse, cumulative
        )
        return output

    @staticmethod
    def backward(ctx: Any, grad):
        from gleipnir.nvidia_mxfp8_varlen_quantize import quantize
        from gleipnir.nvidia_mxfp8_varlen_repack import (
            backward_scales,
            singleton_backward,
        )

        qr, qc, kr, kc, vr, sqr, sqc, skr, skc, svr, output, lse, cu = ctx.saved_tensors
        grad = grad.contiguous()
        dr, sdr, _ = quantize(grad, cu, ctx.maximum, False)
        dc, sdc, _ = quantize(grad, cu, ctx.maximum, True)
        total, hq, _ = qr.shape
        hk, batch = kr.shape[1], cu.numel() - 1
        scales = backward_scales(
            sqr, sqc, skr, skc, svr, sdr, sdc, ctx.maximum, batch, hq, hk, cu
        )
        dq, dk, dv = (
            torch.empty_like(output),
            torch.empty_like(grad[:, :hk]),
            torch.empty_like(grad[:, :hk]),
        )
        # Explicit conversion: the pinned forward emits TH1 statistics whereas
        # native backward reads head-major statistics. Included in all timing.
        stats = lse.transpose(0, 1).contiguous()
        workspace_bytes = batch * hq * ((ctx.maximum + 7) // 8 * 8) * 2 * 4
        # dK/dV reads prologue padding within its final Q tile. A poisoned
        # scratch canary proves these slots require initialization. The unused
        # FP32 dQ accumulator is never accessed by either pinned kernel.
        workspace = torch.zeros(workspace_bytes, device=grad.device, dtype=torch.uint8)
        _, fn = backward_plan(hq, hk)
        tensors = (
            qr,
            kr,
            vr,
            output,
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
        fn(
            tuple(t.data_ptr() for t in tensors),
            total,
            ctx.maximum,
            batch,
            ctx.scale,
            torch.cuda.current_stream().cuda_stream,
        )
        singleton_backward(grad, dq, dk, dv, cu)
        return dq, dk, dv, None, None, None


def packed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative: torch.Tensor,
    maximum: int,
    *,
    scale: float = 0.0625,
) -> torch.Tensor:
    """Run one native attention chain for an entire packed physical row."""
    return _PackedMxfp8.apply(q, k, v, cumulative, maximum, scale)


def packed_mxfp8_interface(original, kernel=packed_attention):
    """Route Transformers' packed row to a single native THD attention call."""

    def attention(module, query, key, value, attention_mask, **kwargs):
        cuts = kwargs.get("cu_seq_lens_q")
        if cuts is None:
            return original(module, query, key, value, attention_mask, **kwargs)
        if query.shape[0] != 1 or attention_mask is not None:
            raise ValueError("packed MXFP8 requires one unpadded physical row")
        other = kwargs.get("cu_seq_lens_k")
        if other is not cuts:
            raise ValueError("packed MXFP8 requires the shared Q/K boundary tensor")
        maximum = kwargs.get("max_length_q")
        if maximum != kwargs.get("max_length_k"):
            raise ValueError("packed MXFP8 requires matching Q/K maximum lengths")
        if kwargs.get("dropout", 0) != 0:
            raise ValueError("packed MXFP8 requires zero attention dropout")
        operands = [x[0].transpose(0, 1).contiguous() for x in (query, key, value)]
        output = kernel(*operands, cuts, maximum, scale=kwargs.get("scaling", 0.0625))
        return output.unsqueeze(0), None

    return attention
