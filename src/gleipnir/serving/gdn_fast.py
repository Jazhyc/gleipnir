"""Leaner host path for vLLM 0.31 Qwen GDN prefill with unchanged kernels.

For prefill-only, non-speculative batches on the FlashInfer GDN backend, each
layer calls the same causal conv1d, fused post-conv, FlashInfer chunked delta
rule and gated RMS norm kernels with the same inputs as upstream, but:

- per-step metadata (int64 `cu_seqlens`, whether every prefill request starts
  from empty state) is computed once per forward, not once per layer;
- FlashInfer writes its output straight into `core_attn_out` (no copy);
- when no request has prior state, FlashInfer starts from zero state and writes
  final states directly into the state pool via `state_indices`, replacing the
  gather / zero-mask / scatter round trip;
- vLLM's CustomOp and wrapper layers around the FlashInfer call are skipped.

Any other batch (decodes, speculative decoding, other backends, profiling) uses
the original method. `set_enabled` supports paired in-server A/B tests.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any

import torch

_STATE: dict[str, Any] = {
    "enabled": True,
    # Bitwise-exact strided gated norm (norm_probe01, norm01); off until adopted.
    "strided_norm": False,
    "step": None,
    "fast": 0,
    "fallback": 0,
    "strided_norms": 0,
}
_NORM_WEIGHTS: dict[int, torch.Tensor] = {}


def set_enabled(enabled: bool, strided_norm: bool | None = None) -> dict:
    _STATE["enabled"] = bool(enabled)
    if strided_norm is not None:
        _STATE["strided_norm"] = bool(strided_norm)
    return audit()


def audit() -> dict:
    keys = ("enabled", "strided_norm", "fast", "fallback", "strided_norms")
    return {k: _STATE[k] for k in keys}


def strided_gated_norm(
    layer: Any, x: torch.Tensor, gate: torch.Tensor, out: torch.Tensor
) -> bool:
    """Upstream gated RMS norm without copying the strided output gate.

    Upstream flattens x/gate/out to [tokens * heads, head_dim], which forces a
    copy of the gate (a strided slice of the packed projection). This launches
    the same FLA Triton kernel over [tokens, heads * head_dim] rows with
    group_size = head_dim, the per-head weight tiled across groups and the
    ROWS_PER_BLOCK upstream would pick for tokens * heads rows, so each
    (token, head) row is reduced with the same tile shape and order.
    """
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
        _LAYER_NORM_FWD_KERNEL,
        calc_rows_per_block,
    )

    norm = layer.norm
    tokens, heads, dim = x.shape
    group = dim if norm.group_size is None else norm.group_size
    if (
        group != dim
        or norm.bias is not None
        or gate.shape != (tokens, heads * dim)
        or gate.stride(-1) != 1
        or not x.is_contiguous()
        or out.data_ptr() != x.data_ptr()
        or dim > 32768
    ):
        return False
    weight = _NORM_WEIGHTS.get(id(layer))
    if weight is None or weight.data_ptr() == 0:
        weight = norm.weight.contiguous().repeat(heads)
        _NORM_WEIGHTS[id(layer)] = weight
    rows_per_block = calc_rows_per_block(tokens * heads, x.device)
    rstd = torch.empty((heads * tokens,), dtype=torch.float32, device=x.device)
    flat = x.view(tokens, heads * dim)
    block_n = 1 << (dim - 1).bit_length()
    _LAYER_NORM_FWD_KERNEL.kernel[
        ((tokens + rows_per_block - 1) // rows_per_block, heads)
    ](
        flat,
        flat,
        weight,
        None,
        gate,
        None,
        rstd,
        flat.stride(0),
        flat.stride(0),
        gate.stride(0),
        tokens,
        dim,
        norm.eps,
        BLOCK_N=block_n,
        ROWS_PER_BLOCK=rows_per_block,
        HAS_BIAS=False,
        HAS_Z=True,
        NORM_BEFORE_GATE=norm.norm_before_gate,
        IS_RMS_NORM=True,
        ACTIVATION=norm.activation,
        num_warps=min(max(block_n // 256, 1), 8),
    )
    _STATE["strided_norms"] += 1
    return True


def step_metadata(
    attn_metadata: Any, fresh: Callable[[int], bool]
) -> tuple[torch.Tensor, bool]:
    """Shared by all GDN layers in one forward (same metadata object)."""
    cached = _STATE["step"]
    if cached is not None and cached[0] is attn_metadata:
        return cached[1], cached[2]
    cu_seqlens = attn_metadata.prefill_query_start_loc.to(torch.int64)
    all_fresh = fresh(attn_metadata.num_prefills)
    _STATE["step"] = (attn_metadata, cu_seqlens, all_fresh)
    return cu_seqlens, all_fresh


def fast_core(
    layer: Any,
    mixed_qkvz: torch.Tensor,
    ba: torch.Tensor,
    core_attn_out: torch.Tensor,
    attn_metadata: Any,
    fresh: Callable[[int], bool],
    kernels: Any,
) -> None:
    """Upstream `_forward_core_fused_norm_packed` for prefill-only batches."""
    qkv_size = (layer.key_dim * 2 + layer.value_dim) // layer.tp_size
    mixed_qkv, output_gate_flat = mixed_qkvz.split(
        [qkv_size, layer.value_dim // layer.tp_size], dim=-1
    )
    output_gate = output_gate_flat.reshape(
        output_gate_flat.size(0), -1, layer.head_v_dim
    )
    b, a = layer.split_ba(ba)
    b = b.contiguous()
    a = a.contiguous()
    conv_state = layer.kv_cache[0]
    if not kernels.is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2)
    ssm_state = layer.kv_cache[1]
    tokens = attn_metadata.num_actual_tokens
    mixed_qkv = mixed_qkv[:tokens]
    b = b[:tokens]
    a = a[:tokens]
    conv_weights = layer.conv1d.weight.view(
        layer.conv1d.weight.size(0), layer.conv1d.weight.size(2)
    )
    conv_output = kernels.causal_conv1d_fn(
        mixed_qkv.transpose(0, 1),
        conv_weights,
        layer.conv1d.bias,
        activation=layer.activation,
        conv_states=conv_state,
        has_initial_state=attn_metadata.has_initial_state,
        cache_indices=attn_metadata.non_spec_state_indices_tensor,
        query_start_loc=attn_metadata.non_spec_query_start_loc,
        metadata=attn_metadata,
    ).transpose(0, 1)
    query, key, value, g, beta = kernels.fused_post_conv_prep(
        conv_output=conv_output,
        a=a,
        b=b,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        num_k_heads=layer.num_k_heads // layer.tp_size,
        head_k_dim=layer.head_k_dim,
        head_v_dim=layer.head_v_dim,
        apply_l2norm=True,
        output_g_exp=False,
    )
    cu_seqlens, all_fresh = step_metadata(attn_metadata, fresh)
    indices = attn_metadata.prefill_state_indices
    # Upstream applies the same conversions in vLLM's FlashInfer wrapper.
    arguments = dict(
        q=query.contiguous(),
        k=key.contiguous(),
        v=value.contiguous(),
        g=torch.exp(g.contiguous().to(torch.float32)),
        beta=beta.contiguous().to(torch.float32),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        output=core_attn_out[:tokens],
        backend="flashinfer",
    )
    from flashinfer import gdn_prefill

    if all_fresh and ssm_state.dtype == torch.float32:
        gdn_prefill.chunk_gated_delta_rule(
            **arguments,
            initial_state=None,
            output_state=ssm_state,
            state_indices=indices,
        )
    else:
        initial_state = ssm_state[indices]
        initial_state[~attn_metadata.prefill_has_initial_state, ...] = 0
        _, final_state = gdn_prefill.chunk_gated_delta_rule(
            **arguments, initial_state=initial_state.to(torch.float32)
        )
        ssm_state[indices] = final_state.to(ssm_state.dtype)
    if not (
        _STATE["strided_norm"]
        and tokens > 0
        and strided_gated_norm(
            layer,
            core_attn_out[:tokens],
            output_gate_flat[:tokens],
            core_attn_out[:tokens],
        )
    ):
        layer._rms_norm_gated_cuda(
            core_attn_out[:tokens], output_gate[:tokens], core_attn_out[:tokens]
        )


def eligible(layer: Any, attn_metadata: Any) -> bool:
    return (
        attn_metadata is not None
        and attn_metadata.num_prefills > 0
        and attn_metadata.num_decodes == 0
        and attn_metadata.spec_sequence_masks is None
        and attn_metadata.prefill_state_indices is not None
        and attn_metadata.prefill_has_initial_state is not None
        and layer.chunk_gated_delta_rule.gdn_prefill_backend == "flashinfer"
    )


def install(layers: list[Any], fresh: Callable[[int], bool]) -> None:
    """Replace each layer's packed fused-norm core; keep the original fallback."""
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    kernels = types.SimpleNamespace(
        causal_conv1d_fn=gdn.causal_conv1d_fn,
        fused_post_conv_prep=gdn.fused_post_conv_prep,
        is_conv_state_dim_first=gdn.is_conv_state_dim_first,
    )
    for layer in layers:
        if getattr(layer, "_gleipnir_gdn_fast", False):
            raise ValueError("GDN fast path is already installed")
        original = layer._forward_core_fused_norm_packed

        def forward(mixed_qkvz, ba, core_attn_out, *, _layer=layer, _orig=original):
            metadata = get_forward_context().attn_metadata
            metadata = (
                metadata.get(_layer.prefix) if isinstance(metadata, dict) else None
            )
            if _STATE["enabled"] and eligible(_layer, metadata):
                _STATE["fast"] += 1
                return fast_core(
                    _layer, mixed_qkvz, ba, core_attn_out, metadata, fresh, kernels
                )
            _STATE["fallback"] += 1
            return _orig(mixed_qkvz, ba, core_attn_out)

        layer._forward_core_fused_norm_packed = forward
        layer._gleipnir_gdn_fast = True
