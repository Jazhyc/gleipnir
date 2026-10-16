"""Run the final full-attention decoder layer only at packed decision positions.

The monitoring loss reads one hidden state per packed example: the last token
of each sequence after the final decoder layer. That layer must still produce
keys and values for every token, but its gated query projection, attention
queries, output projection, MLP and norms are needed only at those decision
positions. Other rows of the returned hidden state are zero; the LM head
reads only decision positions (``logits_to_keep``), so they never reach the
loss. Gradients are mathematically unchanged; arithmetic differs only by
kernel shape (one query per segment instead of the whole segment).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from gleipnir.training.packing import host_cuts, same_boundaries

ATTRIBUTE = "_gleipnir_last_layer_trim"


def _final_layer(model: Any) -> tuple[int, Any]:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = getattr(getattr(base, "model", None), "layers", None)
    layer_types = getattr(base.config, "layer_types", None)
    if layers is None or layer_types is None or len(layers) != len(layer_types):
        raise ValueError("last-layer trim requires Qwen layer metadata")
    if layer_types[-1] != "full_attention" or not hasattr(layers[-1], "self_attn"):
        raise ValueError("last-layer trim requires a full-attention final layer")
    return len(layers) - 1, layers[-1]


@torch.compiler.disable
def _decision_attention(
    attn: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cumulative: torch.Tensor,
) -> torch.Tensor:
    """One query per segment through the untrimmed router's SDPA entry point.

    Each decision query is its sequence's last token, so it attends to every
    key of its own sequence (non-causal for q=1). Kept out of compilation, as
    the untrimmed segmented router is.
    """
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    cuts = host_cuts(cumulative)
    rows = []
    for index, (start, end) in enumerate(zip(cuts[:-1], cuts[1:], strict=True)):
        row, _ = sdpa_attention_forward(
            attn,
            query[:, :, index : index + 1],
            key[:, :, start:end],
            value[:, :, start:end],
            None,
            dropout=0.0,
            scaling=attn.scaling,
            is_causal=True,
        )
        rows.append(row)
    # Each row is (batch, query, heads, head_dim), as the untrimmed path returns.
    return torch.cat(rows, dim=1)


def trimmed_layer_core(
    layer: Any,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    cumulative: torch.Tensor,
) -> torch.Tensor:
    """Layer body for packed decision positions; the compilation target."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    attn = layer.self_attn
    tokens, head_dim = hidden_states.shape[1], attn.head_dim
    selected = positions.shape[0]
    normed = layer.input_layernorm(hidden_states)
    query, gate = torch.chunk(
        attn.q_proj(normed.index_select(1, positions)).view(
            1, selected, -1, head_dim * 2
        ),
        2,
        dim=-1,
    )
    gate = gate.reshape(1, selected, -1)
    query = attn.q_norm(query).transpose(1, 2)
    key = attn.k_norm(attn.k_proj(normed).view(1, tokens, -1, head_dim)).transpose(1, 2)
    value = attn.v_proj(normed).view(1, tokens, -1, head_dim).transpose(1, 2)
    query = apply_rotary_pos_emb(
        query, query, cos.index_select(1, positions), sin.index_select(1, positions)
    )[0]
    key = apply_rotary_pos_emb(key, key, cos, sin)[1]
    output = _decision_attention(attn, query, key, value, cumulative)
    output = output.reshape(1, selected, -1) * torch.sigmoid(gate)
    hidden = hidden_states.index_select(1, positions) + attn.o_proj(output)
    hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden_states.new_zeros(hidden_states.shape).index_copy(1, positions, hidden)


def trimmed_layer_forward(
    layer: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Any = None,
    position_ids: Any = None,
    past_key_values: Any = None,
    *,
    core: Callable[..., torch.Tensor] = trimmed_layer_core,
    **kwargs: Any,
) -> torch.Tensor:
    cumulative = kwargs.get("cu_seq_lens_q")
    sequence_ids = kwargs.get("seq_idx")
    if (
        cumulative is None
        or sequence_ids is None
        or hidden_states.shape[0] != 1
        or attention_mask is not None
        or past_key_values is not None
        or not same_boundaries(cumulative, kwargs["cu_seq_lens_k"])
    ):
        raise ValueError("last-layer trim requires the packed training path")
    if host_cuts(cumulative)[-1] != hidden_states.shape[1]:
        raise ValueError("packed boundaries do not cover the final layer input")
    positions = cumulative[1:].long() - 1
    cos, sin = position_embeddings
    return core(layer, hidden_states, cos, sin, positions, cumulative)


class LastLayerTrim:
    """Switchable wrapper; disabled calls the existing (compiled) forward."""

    def __init__(
        self,
        model: Any,
        compile_function: Callable[[Callable], Callable] | None = None,
    ) -> None:
        """``compile_function`` should match the selective policy's settings."""
        self.layer_index, self.layer = _final_layer(model)
        if getattr(self.layer, ATTRIBUTE, None) is not None:
            raise ValueError("last-layer trim is already installed")
        self.original: Callable[..., torch.Tensor] = self.layer.forward
        self.enabled = True
        self.calls = 0
        self.unpacked_calls = 0
        self.compiled = compile_function is not None
        core = (
            trimmed_layer_core
            if compile_function is None
            else compile_function(trimmed_layer_core)
        )

        def forward(*args: Any, **kwargs: Any) -> torch.Tensor:
            if not self.enabled:
                return self.original(*args, **kwargs)
            if kwargs.get("cu_seq_lens_q") is None:
                # Unpacked diagnostics (e.g. singleton references) stay exact.
                self.unpacked_calls += 1
                return self.original(*args, **kwargs)
            self.calls += 1
            return trimmed_layer_forward(self.layer, *args, core=core, **kwargs)

        self.layer.forward = forward
        setattr(self.layer, ATTRIBUTE, self)

    def metadata(self) -> dict:
        return {
            "enabled": self.enabled,
            "layer_index": self.layer_index,
            "trimmed_calls": self.calls,
            "unpacked_untrimmed_calls": self.unpacked_calls,
            "attention": "per_segment_sdpa_decision_queries",
            "compiled": self.compiled,
        }


def gradient_parity(
    model: Any,
    trim: LastLayerTrim,
    loss_forward: Callable[[], torch.Tensor],
    *,
    capture: Callable[[], torch.Tensor],
) -> dict:
    """Compare loss, decision logits and trainable gradients, trim off vs. on.

    ``loss_forward`` runs one packed forward and returns the loss; ``capture``
    returns the decision logits recorded by that forward. Gradients are
    cleared before and after; no optimizer state is touched.
    """
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    def run(enabled: bool) -> tuple[float, torch.Tensor, list[torch.Tensor]]:
        model.zero_grad(set_to_none=True)
        trim.enabled = enabled
        loss = loss_forward()
        loss.backward()
        grads = [
            torch.zeros_like(p, dtype=torch.float32)
            if p.grad is None
            else p.grad.detach().float().clone()
            for _, p in parameters
        ]
        return float(loss.detach()), capture().detach().float().clone(), grads

    try:
        reference = run(False)
        repeat = run(False)
        candidate = run(True)
    finally:
        trim.enabled = True
        model.zero_grad(set_to_none=True)

    def compare(a, b) -> dict:
        errors = [(x - y).norm() for x, y in zip(a[2], b[2], strict=True)]
        norms = [x.norm() for x in a[2]]
        total = torch.stack(norms).norm()
        relative = [
            (float(e / n) if n > 0 else float(e > 0), name)
            for e, n, (name, _) in zip(errors, norms, parameters, strict=True)
        ]
        worst = max(relative)
        return {
            "loss_abs": abs(a[0] - b[0]),
            "logit_max_abs": float((a[1] - b[1]).abs().max()),
            "gradient_relative_l2": float(torch.stack(errors).norm() / total),
            "worst_parameter_relative_l2": worst[0],
            "worst_parameter": worst[1],
        }

    final_layer = f"layers.{trim.layer_index}."
    # Zero-initialized LoRA B leaves A without gradient in both modes; flag only
    # parameters the untrimmed reference trains that the trim does not.
    missing = [
        name
        for (name, _), ref, grad in zip(
            parameters, reference[2], candidate[2], strict=True
        )
        if final_layer in name
        and bool(ref.abs().sum() > 0)
        and not bool(grad.abs().sum() > 0)
    ]
    finite = all(bool(torch.isfinite(g).all()) for g in candidate[2]) and bool(
        torch.isfinite(candidate[1]).all()
    )
    return {
        "reference_loss": reference[0],
        "noise_floor": compare(reference, repeat),
        "trim_vs_reference": compare(reference, candidate),
        "final_layer_parameters_without_gradient": missing,
        "finite": finite,
    }
