"""LoftQ initialization for Qwen3.5's causal compatibility wrapper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def causal_to_checkpoint_weight(name: str) -> str:
    """Map a causal-model weight name to the native Qwen3.5 checkpoint."""
    if name.startswith("model."):
        return "model.language_model." + name.removeprefix("model.")
    if name.startswith("lm_head."):
        return name
    raise ValueError(f"unsupported Qwen3.5 causal weight name: {name}")


def initialize_qwen35_loftq(model: Any, snapshot: Path) -> int:
    """Replace 4-bit LoRA weights using native multimodal checkpoint tensors."""
    from peft.tuners.lora import Linear4bit
    from peft.utils.loftq_utils import _loftq_init_new
    from safetensors import safe_open

    snapshot = snapshot.resolve()
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing Qwen3.5 weight index: {index_path}")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    prefix = "base_model.model."
    handles: dict[Path, Any] = {}
    initialized = 0
    try:
        for name, module in model.named_modules():
            if not isinstance(module, Linear4bit):
                continue
            if not name.startswith(prefix):
                raise TypeError(f"unexpected PEFT module name: {name}")
            causal_name = name.removeprefix(prefix) + ".weight"
            checkpoint_name = causal_to_checkpoint_weight(causal_name)
            shard_name = weight_map.get(checkpoint_name)
            if shard_name is None:
                raise KeyError(f"Qwen3.5 checkpoint lacks {checkpoint_name}")
            shard_path = snapshot / shard_name
            handle = handles.get(shard_path)
            if handle is None:
                handle = safe_open(shard_path, framework="pt", device="cpu")
                handles[shard_path] = handle
            tensor = handle.get_tensor(checkpoint_name)
            adapter_name = module.active_adapters[0]
            lora_a, lora_b = _loftq_init_new(
                module.weight,
                tensor,
                num_bits=4,
                reduced_rank=module.r[adapter_name],
            )
            module.lora_A[adapter_name].weight.data = lora_a
            module.lora_B[adapter_name].weight.data = lora_b
            initialized += 1
    finally:
        handles.clear()
    if not initialized:
        raise ValueError("LoftQ matched no Qwen3.5 4-bit LoRA modules")
    return initialized
