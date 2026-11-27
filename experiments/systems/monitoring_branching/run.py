"""Bounded CPU mixed-attention Qwen branching score/gradient canary."""

import json

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from gleipnir.branch_training import branched_decision_logits, plan_branches


def canary(
    *,
    checkpoint_segments: bool = False,
    fp32_head: bool = False,
    independent_endpoint: bool = False,
    frozen_embeddings: bool = False,
) -> dict:
    """Compare shared and independent gradients on the same random tiny model."""
    torch.manual_seed(42)
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 2],
        },
    )
    config._attn_implementation = "eager"
    model = Qwen3_5ForCausalLM(config).train()
    if frozen_embeddings:
        model.get_input_embeddings().weight.requires_grad_(False)
    embedding_hooks = tuple(model.get_input_embeddings()._forward_hooks)
    # Explicitly bounded CPU reference, never a production kernel fallback.
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule,
    )

    for layer in model.model.layers:
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.causal_conv1d_fn = None
            layer.linear_attn.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
    trunk = list(range(2, 30))
    requests = [
        trunk[:8] + [40, 41, 42],
        trunk[:9] + [43, 44],
        trunk[:19] + [40, 41, 42],
        trunk + [40, 41, 42],
    ]
    plan = plan_branches(requests)
    targets = torch.tensor([0.2, 0.6, 0.8, 0.9])
    weights = torch.tensor([0.5 / 3] * 3 + [1.0]) / 1.5

    def loss(logits):
        return (
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, 1] - logits[:, 0], targets, reduction="none"
            )
            * weights
        ).sum()

    reference = torch.stack(
        [
            model(torch.tensor([row]), use_cache=False, logits_to_keep=1).logits[
                0, -1, [0, 1]
            ]
            for row in requests
        ]
    )
    loss(reference).backward()
    gradients = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    actual = branched_decision_logits(
        model,
        plan,
        [0, 1],
        checkpoint_segments=checkpoint_segments,
        fp32_head=fp32_head,
        independent_endpoint=independent_endpoint,
    )
    loss(actual).backward()
    assert tuple(model.get_input_embeddings()._forward_hooks) == embedding_hooks
    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-4)
    errors = []
    for name, parameter in model.named_parameters():
        if name in gradients:
            assert parameter.grad is not None, name
            torch.testing.assert_close(
                parameter.grad, gradients[name], atol=2e-5, rtol=2e-3, msg=name
            )
            errors.append((parameter.grad - gradients[name]).abs().max().item())
    return {
        "passed": True,
        "max_logit_error": (actual - reference).abs().max().item(),
        "max_gradient_error": max(errors),
        "independent_tokens": sum(map(len, requests)),
        "branched_tokens": plan.token_work(independent_endpoint=independent_endpoint),
        "gpu_validated": False,
    }


if __name__ == "__main__":
    print(json.dumps(canary(), indent=2))
