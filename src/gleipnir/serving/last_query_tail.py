"""Run the final decoder layer's post-attention work only on pooled rows.

The final layer's output projection, post-attention norm and MLP matter only at
rows the LAST pooler reads. This replaces that tail with one custom op that runs
eagerly on the rows published by last-query attention and returns full-size
tensors (zero hidden state elsewhere). The op is registered as a vLLM splitting
op, like attention, so CUDA graphs never bake in a row selection.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch

from gleipnir.serving.last_query_attention import LastQueryState

OP_NAME = "gleipnir_last_query_tail"
SPLITTING_OP = f"vllm::{OP_NAME}"


@dataclasses.dataclass
class _Entry:
    layer: torch.nn.Module
    state: LastQueryState
    current_context: Callable[[], object]
    calls: int = 0
    rows: int = 0


_LAYERS: dict[str, _Entry] = {}


def tail(
    layer: torch.nn.Module,
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    rows: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The original gated output, norm and MLP on `rows` (all rows if None)."""
    if rows is None:
        # Fused norms may update their residual in place; never alias inputs.
        hidden, _ = layer.self_attn.o_proj(attn_output * torch.sigmoid(gate))
        hidden, residual = layer.post_attention_layernorm(hidden, residual.clone())
        return layer.mlp(hidden), residual
    hidden_out = torch.zeros_like(residual)
    residual_out = residual.clone()
    if rows.numel():
        selected = attn_output.index_select(0, rows) * torch.sigmoid(
            gate.index_select(0, rows)
        )
        hidden, _ = layer.self_attn.o_proj(selected)
        hidden, kept = layer.post_attention_layernorm(
            hidden, residual.index_select(0, rows)
        )
        hidden_out.index_copy_(0, rows, layer.mlp(hidden))
        residual_out.index_copy_(0, rows, kept)
    return hidden_out, residual_out


def last_query_tail(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    entry = _LAYERS[layer_name]
    context = entry.current_context()
    if getattr(context, "attn_metadata", None) is None:
        # Profiling and dummy runs have no batch: keep the full original work.
        return tail(entry.layer, attn_output, gate, residual, None)
    if entry.state.context is not context or entry.state.rows is None:
        raise ValueError("last-query rows were not published for this forward")
    entry.calls += 1
    entry.rows += int(entry.state.rows.numel())
    return tail(entry.layer, attn_output, gate, residual, entry.state.rows)


def _fake(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(residual), torch.empty_like(residual)


def register() -> None:
    """Register the op once and add it to vLLM's default splitting ops."""
    from vllm.config.compilation import CompilationConfig
    from vllm.utils.torch_utils import direct_register_custom_op

    if not hasattr(torch.ops.vllm, OP_NAME):
        direct_register_custom_op(
            op_name=OP_NAME,
            op_func=last_query_tail,
            mutates_args=[],
            fake_impl=_fake,
        )
    if SPLITTING_OP not in CompilationConfig._attention_ops:
        CompilationConfig._attention_ops.append(SPLITTING_OP)


def install(
    decoder_layer: torch.nn.Module,
    name: str,
    state: LastQueryState,
    current_context: Callable[[], object],
) -> None:
    """Replace one gated full-attention decoder layer's forward before compile."""
    attention = decoder_layer.self_attn
    if (
        decoder_layer.layer_type != "full_attention"
        or decoder_layer.layer_scale
        or decoder_layer.use_attn_reduce_scatter_for_moe
        or not attention.attn_output_gate
    ):
        raise ValueError("last-query tail requires a plain gated attention layer")
    if name in _LAYERS:
        raise ValueError("last-query tail is already installed")
    register()
    _LAYERS[name] = _Entry(decoder_layer, state, current_context)
    op = getattr(torch.ops.vllm, OP_NAME)

    def forward(hidden_states, residual, positions, **kwargs):
        if residual is None:
            raise ValueError("final-layer tail requires an incoming residual")
        hidden_states, residual = decoder_layer.input_layernorm(hidden_states, residual)
        qkv, _ = attention.qkv_proj(hidden_states)
        q, k, v, gate = attention._project_qkv_gate(qkv, positions)
        attn_output = attention.attn(q, k, v)
        return op(attn_output, gate, residual, name)

    decoder_layer.forward = forward


def audit(name: str) -> dict:
    entry = _LAYERS[name]
    return {"calls": entry.calls, "rows": entry.rows}
