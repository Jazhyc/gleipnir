"""Explicit bounded training integration of selected Meta-inspired variants."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from unittest.mock import patch

import torch

from gleipnir.nvidia_mxfp8_meta_variants import NativeVariant, native_variant


def validate_options(options: dict) -> tuple[bool, NativeVariant]:
    """Reject misspelled or implicit precision choices before model execution."""
    allowed = {"norm_rope", *asdict(NativeVariant())}
    if set(options) - allowed:
        raise ValueError("unknown Meta attention optimization option")
    norm = options.get("norm_rope", False)
    if type(norm) is not bool:
        raise ValueError("norm_rope must be an explicit boolean")
    variant = NativeVariant(**{k: v for k, v in options.items() if k != "norm_rope"})
    for name in ("persistent_dq", "persistent_dkdv", "ds_warp_amax"):
        if type(getattr(variant, name)) is not bool:
            raise ValueError("native scheduling/scaling options must be boolean")
    return norm, variant


@contextmanager
def meta_attention_runtime(options: dict):
    """Keep unpacked reference calls and restore model methods on all exits."""
    norm, variant = validate_options(options)
    with ExitStack() as stack:
        stack.enter_context(native_variant(variant))
        if norm:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

            from gleipnir.nvidia_mxfp8_norm_rope import norm_rope_attention

            original = Qwen3_5Attention.forward
            kernel = torch.compiler.disable(norm_rope_attention)

            def forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=None,
                **kwargs,
            ):
                cuts = kwargs.get("cu_seq_lens_q")
                if cuts is None:
                    return original(
                        self,
                        hidden_states,
                        position_embeddings,
                        attention_mask,
                        past_key_values,
                        **kwargs,
                    )
                if (
                    hidden_states.shape[0] != 1
                    or self.head_dim != 256
                    or past_key_values is not None
                    or self.attention_dropout != 0
                    or self.q_norm.eps != self.k_norm.eps
                    or self.scaling != 0.0625
                ):
                    raise ValueError(
                        "Meta producer requires packed text-only D256 training"
                    )
                maximum = kwargs.get("max_length_q")
                if maximum != kwargs.get("max_length_k"):
                    raise ValueError(
                        "Meta producer requires matched self-attention cuts"
                    )
                kv_cuts = kwargs.get("cu_seq_lens_k")
                if kv_cuts is not cuts:
                    raise ValueError(
                        "Meta producer requires shared Q/K cumulative storage"
                    )
                total = hidden_states.shape[1]
                slab = self.q_proj(hidden_states).view(total, -1, 512)
                q, gate = slab[..., :256], slab[..., 256:]
                k = self.k_proj(hidden_states).view(total, -1, 256)
                v = self.v_proj(hidden_states).view(total, -1, 256)
                cos, sin = position_embeddings
                out = kernel(
                    q,
                    k,
                    v,
                    self.q_norm.weight,
                    self.k_norm.weight,
                    cos[0].contiguous(),
                    sin[0].contiguous(),
                    cuts,
                    maximum,
                    eps=self.q_norm.eps,
                )
                out = out.view(1, total, -1) * torch.sigmoid(gate.reshape(1, total, -1))
                return self.o_proj(out), None

            stack.enter_context(patch.object(Qwen3_5Attention, "forward", forward))
        yield {"norm_rope": norm, **asdict(variant)}
