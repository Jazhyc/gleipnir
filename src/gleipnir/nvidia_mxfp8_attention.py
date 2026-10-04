"""Experimental dense causal D256 GQA MXFP8 training on Blackwell.

NVIDIA's FROST adapter requires separately quantized orientations. Quantization
and its scale buffers remain local to each sequence; no packed THD claim is made.
Imports are lazy so ordinary FA4 training does not depend on this overlay.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import torch

FRONTEND_VERSION = "1.31.0"
UPSTREAM_REVISION = "51d9d06b574222378a3d806009accab098e73705"
FORWARD_ENGINE = "sdpa_fwd_prefill_sm100_mxfp8"
BACKWARD_ENGINE = "sdpa_bwd_sm100_mxfp8"


def validate_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """Reject inputs outside the bounded dense self-attention contract."""
    if any(x.ndim != 3 for x in (q, k, v)):
        raise ValueError("MXFP8 inputs must have shape [tokens, heads, 256]")
    if (
        q.shape[0] < 1
        or q.shape[0] != k.shape[0]
        or k.shape != v.shape
        or any(x.shape[2] != 256 for x in (q, k, v))
        or k.shape[1] < 1
        or q.shape[1] % k.shape[1]
    ):
        raise ValueError("MXFP8 requires nonempty D256 self attention and integer GQA")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise ValueError("MXFP8 experimental boundary requires BF16 QKV")
    if not (q.device == k.device == v.device):
        raise ValueError("MXFP8 QKV must share a device")


@lru_cache(maxsize=1)
def _cudnn() -> Any:
    import cudnn

    if cudnn.__version__ != FRONTEND_VERSION:
        raise ValueError(f"cuDNN Frontend version drift: {cudnn.__version__}")
    if torch.cuda.get_device_capability() != (10, 0):
        raise ValueError("this MXFP8 screen requires the B200 SM100 recipe")
    return cudnn


def _dims(heads: int, length: int) -> list[int]:
    return [1, heads, length, 256]


def _strides(heads: int, length: int) -> list[int]:
    return [length * heads * 256, 256, heads * 256, 1]


def _sf_dims(heads: int, length: int, columnwise: bool) -> list[int]:
    padded = (length + 127) // 128 * 128
    return [1, heads, padded // 32, 256] if columnwise else [1, heads, padded, 8]


def _sf(graph: Any, name: str, heads: int, length: int, col: bool) -> Any:
    c = _cudnn()
    dims = _sf_dims(heads, length, col)
    return graph.tensor(
        name=name,
        dim=dims,
        stride=[dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1],
        data_type=c.data_type.FP8_E8M0,
        reordering_type=c.tensor_reordering.F8_128x4,
    )


def _select(graph: Any, engine: str) -> None:
    c = _cudnn()
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([c.heur_mode.A])
    names = [
        graph.get_plan_name_at_index(i) for i in range(graph.get_execution_plan_count())
    ]
    indices = [
        i
        for i, name in enumerate(names)
        if name == engine or name.startswith(engine + "[")
    ]
    if not indices:
        raise RuntimeError(f"required engine {engine} was not offered: {names}")
    graph.select_plan(indices[0])
    graph.check_support()
    graph.build_plans()


@lru_cache(maxsize=512)
def _plan(
    length: int, hq: int, hkv: int, scale: float, backward: bool, device: int
) -> tuple[Any, dict]:
    c = _cudnn()
    g = c.pygraph(
        io_data_type=c.data_type.FP8_E4M3,
        intermediate_data_type=c.data_type.FLOAT,
        compute_data_type=c.data_type.FLOAT,
    )
    t = {}

    def payload(name: str, heads: int, dtype: Any = c.data_type.FP8_E4M3) -> Any:
        t[name] = g.tensor(
            name=name,
            dim=_dims(heads, length),
            stride=_strides(heads, length),
            data_type=dtype,
        )
        return t[name]

    for name, heads in (("q", hq), ("k", hkv), ("v", hkv)):
        payload(name, heads)
        t["sf_" + name] = _sf(
            g, "sf_" + name, heads, length, name == "v" and not backward
        )
    if backward:
        for name, heads in (("q_T", hq), ("k_T", hkv), ("do", hq), ("do_T", hq)):
            payload(name, heads)
            t["sf_" + name] = _sf(g, "sf_" + name, heads, length, name.endswith("_T"))
        payload("o", hq, c.data_type.BFLOAT16)
        payload("do_f16", hq, c.data_type.BFLOAT16)
        t["stats"] = g.tensor(
            name="stats",
            dim=[1, hq, length, 1],
            stride=[hq * length, length, 1, 1],
            data_type=c.data_type.FLOAT,
        )
        dq, dk, dv, *amax = g.sdpa_mxfp8_backward(
            q=t["q"],
            q_T=t["q_T"],
            k=t["k"],
            k_T=t["k_T"],
            v=t["v"],
            o_f16=t["o"],
            dO_f16=t["do_f16"],
            dO=t["do"],
            dO_T=t["do_T"],
            stats=t["stats"],
            descale_q=t["sf_q"],
            descale_q_T=t["sf_q_T"],
            descale_k=t["sf_k"],
            descale_k_T=t["sf_k_T"],
            descale_v=t["sf_v"],
            descale_dO=t["sf_do"],
            descale_dO_T=t["sf_do_T"],
            attn_scale=scale,
            use_causal_mask=True,
        )
        for name, out, heads in (("dq", dq, hq), ("dk", dk, hkv), ("dv", dv, hkv)):
            t[name] = (
                out.set_output(True)
                .set_data_type(c.data_type.BFLOAT16)
                .set_dim(_dims(heads, length))
                .set_stride(_strides(heads, length))
            )
        for a in amax:
            a.set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(
                c.data_type.FLOAT
            )
    else:
        o, stats, _amax = g.sdpa_mxfp8(
            q=t["q"],
            k=t["k"],
            v=t["v"],
            descale_q=t["sf_q"],
            descale_k=t["sf_k"],
            descale_v=t["sf_v"],
            attn_scale=scale,
            generate_stats=True,
            use_causal_mask=True,
        )
        t["o"] = (
            o.set_output(True)
            .set_dim(_dims(hq, length))
            .set_stride(_strides(hq, length))
            .set_data_type(c.data_type.BFLOAT16)
        )
        t["stats"] = (
            stats.set_output(True)
            .set_dim([1, hq, length, 1])
            .set_stride([hq * length, length, 1, 1])
            .set_data_type(c.data_type.FLOAT)
        )
    _select(g, BACKWARD_ENGINE if backward else FORWARD_ENGINE)
    return g, t


def quantize(
    source: torch.Tensor, columnwise: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use NVIDIA's fused CUDA quantizer and canonical F8_128x4 scale layout."""
    from cudnn.gated_attention_block.kernels.quantize_mxfp8 import (
        compile_quantize_mxfp8,
        run_quantize_mxfp8,
        sf_bytes,
    )

    source = source.contiguous()
    length, heads, dim = source.shape
    recipe = compile_quantize_mxfp8(
        dtype_in=source.dtype, h=heads, d=dim, axis="col" if columnwise else "row"
    )
    data = torch.empty(source.shape, device=source.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty(
        sf_bytes(1, heads, length, dim), device=source.device, dtype=torch.uint8
    )
    run_quantize_mxfp8(
        recipe,
        source,
        data,
        sf,
        batch=1,
        seq_len=length,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    return data, sf


def _execute(g: Any, descriptors: dict, buffers: dict) -> None:
    workspace = torch.empty(
        max(g.get_workspace_size(), 1), dtype=torch.uint8, device=buffers["q"].device
    )
    # Physical THD payloads are views of logical BHSD over BSHD storage.
    bindings = {
        descriptors[name]: value.unsqueeze(0).transpose(1, 2)
        if value.ndim == 3
        else value
        for name, value in buffers.items()
    }
    g.execute(bindings, workspace)


class _Mxfp8Attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
    ) -> torch.Tensor:
        validate_inputs(q, k, v)
        c = _cudnn()
        del c
        length, hq, _ = q.shape
        hkv = k.shape[1]
        qr, sqr = quantize(q, False)
        qt, sqt = quantize(q, True)
        kr, skr = quantize(k, False)
        kt, skt = quantize(k, True)
        vr, svr = quantize(v, False)
        vc, svc = quantize(v, True)
        o = torch.empty_like(q, memory_format=torch.contiguous_format)
        stats = torch.empty((1, hq, length, 1), device=q.device, dtype=torch.float32)
        g, t = _plan(length, hq, hkv, scale, False, q.device.index)
        _execute(
            g, t, dict(q=qr, k=kr, v=vc, sf_q=sqr, sf_k=skr, sf_v=svc, o=o, stats=stats)
        )
        ctx.save_for_backward(qr, qt, kr, kt, vr, sqr, sqt, skr, skt, svr, o, stats)
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor) -> tuple:
        qr, qt, kr, kt, vr, sqr, sqt, skr, skt, svr, o, stats = ctx.saved_tensors
        do = do.contiguous()
        dor, sdor = quantize(do, False)
        dot, sdot = quantize(do, True)
        dq, dk, dv = [
            torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
            for x in (qr, kr, vr)
        ]
        length, hq, _ = qr.shape
        g, t = _plan(length, hq, kr.shape[1], ctx.scale, True, qr.device.index)
        _execute(
            g,
            t,
            dict(
                q=qr,
                q_T=qt,
                k=kr,
                k_T=kt,
                v=vr,
                sf_q=sqr,
                sf_q_T=sqt,
                sf_k=skr,
                sf_k_T=skt,
                sf_v=svr,
                o=o,
                stats=stats,
                do_f16=do,
                do=dor,
                do_T=dot,
                sf_do=sdor,
                sf_do_T=sdot,
                dq=dq,
                dk=dk,
                dv=dv,
            ),
        )
        return dq, dk, dv, None


def mxfp8_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, scale: float | None = None
) -> torch.Tensor:
    """Dense causal self attention with NVIDIA's approximate MXFP8 backward."""
    validate_inputs(q, k, v)
    if not q.is_cuda:
        raise ValueError("MXFP8 attention requires CUDA")
    scale = 0.0625 if scale is None else scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("MXFP8 softmax scale must be positive and finite")
    return _Mxfp8Attention.apply(q, k, v, scale)


def segmented_mxfp8_interface(
    original: Callable, kernel: Callable = mxfp8_attention
) -> Callable:
    """Run the dense kernel independently for every original packed example."""

    def attention(module, query, key, value, attention_mask, **kwargs):
        cumulative = kwargs.get("cu_seq_lens_q")
        if cumulative is None:
            return original(module, query, key, value, attention_mask, **kwargs)
        if query.shape[0] != 1 or attention_mask is not None:
            raise ValueError("MXFP8 packing requires one unpadded row")
        other = kwargs.get("cu_seq_lens_k")
        if other is None or not torch.equal(cumulative, other):
            raise ValueError("MXFP8 requires matching Q/K sequence boundaries")
        cuts = cumulative.tolist()
        if (
            cumulative.dtype != torch.int32
            or cuts[0] != 0
            or cuts[-1] != query.shape[2]
            or any(b <= a for a, b in zip(cuts[:-1], cuts[1:], strict=True))
        ):
            raise ValueError("invalid MXFP8 packed boundaries")
        if kwargs.get("dropout", 0) != 0:
            raise ValueError("MXFP8 screen requires zero attention dropout")
        parts = []
        for start, end in zip(cuts[:-1], cuts[1:], strict=True):
            operands = [
                x[0, :, start:end].transpose(0, 1).contiguous()
                for x in (query, key, value)
            ]
            parts.append(kernel(*operands, scale=kwargs.get("scaling")))
        return torch.cat(parts).unsqueeze(0), None

    return attention
