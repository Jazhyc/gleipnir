"""Pinned two-device QLoRA model for differentiable prompted branches."""

from pathlib import Path

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_branch_model(*, seed: int, adapter: Path | None = None):
    """Fresh FP32 master adapters by default; never warm-start implicitly."""
    import importlib.metadata

    import torch
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import BitsAndBytesConfig, Qwen3_5ForCausalLM

    if torch.cuda.device_count() != 2:
        raise RuntimeError("branch training requires exactly two visible GPUs")
    for package in ("flash-linear-attention", "fla-core"):
        if importlib.metadata.version(package) != "0.5.2":
            raise RuntimeError(f"unpinned production kernel: {package}")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    placement = {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,
        "model.norm": 1,
        "lm_head": 0,
        **{f"model.layers.{i}": int(i >= 16) for i in range(32)},
    }
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=placement,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if adapter is None:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=128,
                lora_alpha=256,
                lora_dropout=0.0,
                bias="none",
                target_modules=TARGET_MODULES,
            ),
        )
    else:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    kernels = set()
    for layer in model.get_base_model().model.layers:
        if hasattr(layer, "linear_attn"):
            attention = layer.linear_attn
            kernel = attention.chunk_gated_delta_rule.__module__
            if not kernel.startswith("fla.") or attention.causal_conv1d_fn is None:
                raise RuntimeError("production FLA/causal-conv kernels unavailable")
            kernels.add(kernel)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params or any(p.dtype != torch.float32 for p in params):
        raise RuntimeError("master adapters must be FP32")
    metadata = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "device_map": placement,
        "initialization": "fresh_base" if adapter is None else str(adapter),
        "trainable_parameters": sum(p.numel() for p in params),
        "master_dtype": "float32",
        "kernels": sorted(kernels),
        "packages": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "transformers",
                "peft",
                "bitsandbytes",
                "flash-linear-attention",
                "fla-core",
                "causal-conv1d",
                "triton",
            )
        },
        "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
    }
    return model.train(), metadata
