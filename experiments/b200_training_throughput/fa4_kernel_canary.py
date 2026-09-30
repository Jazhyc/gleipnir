"""Bounded BF16 GQA forward/backward check against Torch FP32 math attention."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from flash_attn.cute import flash_attn_func
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoConfig

torch.manual_seed(0)
config = AutoConfig.from_pretrained(
    "Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)
config = getattr(config, "text_config", config)
shape = {
    "heads": config.num_attention_heads,
    "kv_heads": config.num_key_value_heads,
    "head_dim": config.head_dim,
    "tokens": 256,
}
q, k, v = [
    torch.randn(
        1, 256, heads, shape["head_dim"], device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    for heads in (shape["heads"], shape["kv_heads"], shape["kv_heads"])
]
rq, rk, rv = [t.detach().float().requires_grad_() for t in (q, k, v)]
with sdpa_kernel(SDPBackend.MATH):
    reference = F.scaled_dot_product_attention(
        rq.transpose(1, 2),
        rk.transpose(1, 2),
        rv.transpose(1, 2),
        is_causal=True,
        enable_gqa=True,
    ).transpose(1, 2)
candidate = flash_attn_func(q, k, v, causal=True)
gradient = torch.randn_like(candidate)
reference.backward(gradient.float())
candidate.backward(gradient)
errors = {}
for name, actual, expected, limit in (
    ("forward", candidate, reference, 0.02),
    ("dq", q.grad, rq.grad, 0.05),
    ("dk", k.grad, rk.grad, 0.05),
    ("dv", v.grad, rv.grad, 0.05),
):
    assert actual is not None and torch.isfinite(actual).all()
    relative_l2 = float(
        torch.linalg.vector_norm(actual.float() - expected)
        / torch.linalg.vector_norm(expected)
    )
    errors[name] = {"relative_l2": relative_l2, "limit": limit}
    assert relative_l2 <= limit, (name, relative_l2, limit)
report = {"shape": shape, "errors": errors, "passed": True}
Path("logs/runpod/b200_training_throughput_fa4/kernel_canary.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report), flush=True)
