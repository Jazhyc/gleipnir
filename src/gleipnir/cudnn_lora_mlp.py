"""Isolated LoRA-aware cuDNN gate/up/SwiGLU graph with explicit BF16 backward.

Uses the existing pinned cuDNN overlay; no production runtime import is implied.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_PLANS: dict = {}


def _fused(x, wg, wu, rg, ru):
    import cudnn
    from cudnn.gemm.ops.swiglu_mlp import _autotune, _handle

    if any(
        t.dtype != torch.bfloat16 or t.device != x.device for t in (x, wg, wu, rg, ru)
    ):
        raise ValueError("cuDNN LoRA fusion requires BF16 operands on one GPU")
    operands = [
        x.unsqueeze(0),
        wg.t().unsqueeze(0),
        wu.t().unsqueeze(0),
        rg.unsqueeze(0),
        ru.unsqueeze(0),
    ]
    handle, stream = _handle(x.device)
    key = (
        tuple((tuple(t.shape), tuple(t.stride())) for t in operands),
        x.device.index,
        stream,
    )
    outputs = [torch.empty_like(rg) for _ in range(3)]
    buffers = operands + [t.unsqueeze(0) for t in outputs]
    if key not in _PLANS:
        bf, fp = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
        graph = cudnn.pygraph(handle=handle, compute_data_type=fp)
        inputs = [
            graph.tensor(dim=list(t.shape), stride=list(t.stride()), data_type=bf)
            for t in operands
        ]
        X, WG, WU, RG, RU = inputs
        bg = graph.matmul(A=X, B=WG).set_data_type(bf)
        bu = graph.matmul(A=X, B=WU).set_data_type(bf)
        gate = graph.add(a=bg, b=RG).set_data_type(bf).set_output(True)
        up = graph.add(a=bu, b=RU).set_data_type(bf).set_output(True)
        act = graph.swish(input=gate).set_data_type(bf)
        out = graph.mul(a=act, b=up).set_data_type(bf).set_output(True)
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        uids = tuple(t.get_uid() for t in (*inputs, out, gate, up))
        best, ws = _autotune(graph, handle, buffers, uids)
        _PLANS[key] = graph, uids, best, ws
    graph, uids, best, ws = _PLANS[key]
    graph.execute_plan_at_index(
        buffers, ws, index=best, handle=handle, tensor_uids=uids
    )
    return outputs


class _Activation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wg, wu, rg, ru):
        out, gate, up = _fused(x, wg, wu, rg, ru)
        ctx.save_for_backward(wg, wu, gate, up)
        return out

    @staticmethod
    def backward(ctx, dh):
        wg, wu, gate, up = ctx.saved_tensors
        # Exact eager BF16 multiply + SiLU-backward rounding boundaries.
        with torch.autocast("cuda", enabled=False):
            du = dh * F.silu(gate)
            dg = torch.ops.aten.silu_backward(dh * up, gate)
            dx = F.linear(dg, wg.t()) + F.linear(du, wu.t())
        return dx, None, None, dg, du


def cudnn_forward(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Retain ordinary adapter matmuls and down projection with autograd."""
    from gleipnir.mlp_gemm import update

    shape = x.shape
    flat = x.reshape(-1, shape[-1]).contiguous()
    h = _Activation.apply(
        flat,
        module.gate_proj.weight,
        module.up_proj.weight,
        update(module.gate_proj, flat),
        update(module.up_proj, flat),
    )
    return module.down_proj(h.reshape(*shape[:-1], -1))
