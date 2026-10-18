"""Config-derived D256 cuDNN MXFP8 prefill over a BF16 paged serving cache.

The 4B default's ``gleipnir.serving.mxfp8`` is hash-bound and fixes GQA 16/4
in its cuDNN descriptors and call guards. This module reuses its producer
kernels, token-offset kernel and hardware-exp2 source rewrite by import, and
builds the same cuDNN ``SdpaFwdDslSm100`` plan for any query/KV head count
with head dimension 256. Length and batch bounds are read from the original
module at call time, so ``experiments.b200_context_million.envelope.enable``
applies to both.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import torch

from gleipnir.serving import mxfp8 as base
from gleipnir.serving.mxfp8 import _token_offsets, produce
from gleipnir.serving.mxfp8_source import hardware_exp2_source

HEAD_DIM = 256
SCALE = HEAD_DIM**-0.5


@lru_cache(maxsize=8)
def forward_plan(
    device: torch.device, query_heads: int, kv_heads: int, max_length: int
):
    """Compile packed query/history with bottom-right causal alignment."""
    from cudnn.api_base import TensorDesc
    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.fwd import SdpaFwdDslSm100, api_dsl

    from gleipnir.nvidia_mxfp8_attention import _cudnn

    if query_heads % kv_heads:
        raise ValueError("MXFP8 GQA requires query heads divisible by KV heads")
    _cudnn()

    def desc(heads, dtype):
        return TensorDesc(
            dtype,
            (base.MAX_BATCH, heads, max_length, HEAD_DIM),
            (max_length * heads * HEAD_DIM, HEAD_DIM, heads * HEAD_DIM, 1),
            (3, 1, 2, 0),
            device,
        )

    original_loader = api_dsl._load_sm100_kernel_module

    def loader(flavor, params, **kwargs):
        original = original_loader(flavor, params, **kwargs)
        if flavor != (256, 256) or kwargs.get("pertensor") or kwargs.get("rubin"):
            raise ValueError("MXFP8 hardware-exp2 variant requires SM100 D256")
        source = hardware_exp2_source(Path(original.__file__).read_text())
        digest = hashlib.sha256(source.encode()).hexdigest()
        # Same derived-source cache as the 4B module: identical bytes, one file.
        target = (
            Path(__file__).resolve().parents[3]
            / ".cache/kernels/nvidia_mxfp8/generated"
            / ("serving_exp2_" + digest + ".py")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(source)
        elif target.read_text() != source:
            raise ValueError("MXFP8 derived-source identity drift")
        return load_template(str(target), params, tag="gleipnir_serving_exp2_" + digest)

    with patch.object(api_dsl, "_load_sm100_kernel_module", loader):
        plan = SdpaFwdDslSm100(
            desc(query_heads, torch.float8_e4m3fn),
            desc(kv_heads, torch.float8_e4m3fn),
            desc(kv_heads, torch.float8_e4m3fn),
            desc(query_heads, torch.bfloat16),
            is_causal=True,
            causal_bottom_right=True,
            scale_softmax=SCALE,
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


def split_paged_cache(
    cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor], kv_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept the original HND cache or vLLM 0.31's strided K/V views."""
    if isinstance(cache, tuple):
        if len(cache) != 2:
            raise ValueError("MXFP8 requires exactly two K/V cache views")
        key, value = cache
    else:
        if cache.ndim != 5 or cache.shape[1] != 2:
            raise ValueError("MXFP8 requires a combined HND cache")
        key, value = cache.unbind(1)
    if (
        key.ndim != 4
        or key.shape != value.shape
        or key.shape[1] != kv_heads
        or key.shape[-1] != HEAD_DIM
        or key.dtype != torch.bfloat16
        or value.dtype != torch.bfloat16
        or key.device != value.device
        or key.stride(-1) != 1
        or value.stride(-1) != 1
    ):
        raise ValueError("MXFP8 requires matching BF16 HND D256 K/V views")
    return key, value


def paged_forward(
    *,
    query_heads: int,
    kv_heads: int,
    query: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    block_tables: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    cum_seq_lens_kv: torch.Tensor,
    seq_lens: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    batch_size: int,
    out: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Adapt only the audited BF16 causal TRTLLM prefill call; never fall back."""
    key, value = split_paged_cache(kv_cache, kv_heads)
    if (
        query.shape[1:] != (query_heads, HEAD_DIM)
        or query.dtype != torch.bfloat16
        or out.dtype != torch.bfloat16
        or batch_size != cum_seq_lens_q.numel() - 1
        or cum_seq_lens_kv.numel() != batch_size + 1
        or seq_lens.shape != (batch_size,)
        or seq_lens.dtype != torch.int32
    ):
        raise ValueError(
            f"MXFP8 serving requires BF16 causal D256 GQA {query_heads}/{kv_heads}"
            " and HND cache"
        )
    for name, neutral in (
        ("bmm1_scale", SCALE),
        ("bmm2_scale", 1.0),
        ("window_left", -1),
        ("sinks", None),
        ("o_sf_scale", None),
        ("kv_cache_sf", None),
        ("causal", True),
    ):
        if kwargs.get(name, neutral) != neutral:
            raise ValueError(f"unsupported MXFP8 serving option: {name}")
    # TRTLLM's cum_seq_lens_kv counts cache pages. cuDNN's packed THD
    # offsets count tokens; derive them from exact lengths, including tails.
    token_offsets = torch.empty_like(cum_seq_lens_kv)
    _token_offsets[(1,)](seq_lens, token_offsets, batch_size)
    qr, sfq = produce(query, cum_seq_lens_q, max_q_len)
    capacity = batch_size * max_kv_len
    kr, sfk = produce(
        key, token_offsets, max_kv_len, table=block_tables, total=capacity
    )
    vc, sfv = produce(
        value,
        token_offsets,
        max_kv_len,
        table=block_tables,
        total=capacity,
        column=True,
    )
    plan = forward_plan(query.device, query_heads, kv_heads, base.MAX_LENGTH)
    workspace = torch.empty(
        max(1, plan.scratch_workspace_bytes()), device=query.device, dtype=torch.uint8
    )
    plan.execute(
        qr,
        kr,
        vc,
        out,
        seq_q_lens=cum_seq_lens_q,
        seq_kv_lens=token_offsets,
        sf_q=sfq,
        sf_k=sfk,
        sf_v=sfv,
        workspace=workspace,
    )
    return out
