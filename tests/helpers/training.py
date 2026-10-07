"""Shared packed-training configuration for CPU contract checks."""


def student_config():
    return {
        "quantization": {
            "enabled": False,
            "full_bf16_lora": True,
            "mlp_precision": "bf16",
        },
        "model_loader": "causal_lm",
        "finetuning_mode": "lora",
        "attn_implementation": "sdpa",
        "lora": {"dropout": 0},
        "training": {
            "sequence_packing": True,
            "gated_delta_backend": "flashqla",
            "adaptive_microbatching": {"enabled": True, "max_padded_tokens": 256},
            "selective_torch_compile_policy": "full_attention_and_linear_shell",
            "selective_torch_compile_canary_tokens": 256,
        },
    }
