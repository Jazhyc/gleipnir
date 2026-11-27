"""Exact weight average of same-initialization LoRA adapters as one LoRA.

For adapters i = 1..n with rank r, alpha a and shared target modules, the
averaged update is (1/n) sum_i (a/r) B_i A_i. Stacking A_i along the rank and
B_i / n along the input of B gives a rank n*r LoRA whose update is exactly
that average when alpha is n*a (scaling a/r unchanged). The merge, parity and
serving pipeline then runs unchanged.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

WEIGHTS = "adapter_model.safetensors"
CONFIG = "adapter_config.json"


def stack(tensors: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = set(tensors[0])
    if any(set(t) != keys for t in tensors):
        raise ValueError("adapters have different tensor keys")
    count = len(tensors)
    out = {}
    for key in sorted(keys):
        parts = [t[key].float() for t in tensors]
        if ".lora_A." in key:
            out[key] = torch.cat(parts, dim=0)
        elif ".lora_B." in key:
            out[key] = torch.cat([p / count for p in parts], dim=1)
        else:
            raise ValueError(f"non-LoRA tensor cannot be averaged exactly: {key}")
    return out


def averaged_config(configs: list[dict]) -> dict:
    base = configs[0]
    for config in configs[1:]:
        for field in ("r", "lora_alpha", "use_rslora", "use_dora"):
            if config.get(field) != base.get(field):
                raise ValueError(f"adapter config field differs: {field}")
        # PEFT serializes the module set in arbitrary order.
        if set(config.get("target_modules") or []) != set(
            base.get("target_modules") or []
        ):
            raise ValueError("adapter config field differs: target_modules")
    if base.get("use_rslora") or base.get("use_dora"):
        raise ValueError("only plain LoRA scaling averages exactly")
    if any(c.get("rank_pattern") or c.get("alpha_pattern") for c in configs):
        raise ValueError("per-module rank/alpha overrides do not average exactly")
    count = len(configs)
    return {**base, "r": base["r"] * count, "lora_alpha": base["lora_alpha"] * count}


def update(tensors: dict[str, torch.Tensor], module: str, scale: float):
    a = tensors[f"{module}.lora_A.weight"].double()
    b = tensors[f"{module}.lora_B.weight"].double()
    return scale * (b @ a)


def build(sources: list[Path], destination: Path) -> dict:
    """Write the stacked FP32 adapter; return a per-module exactness receipt."""
    from safetensors.torch import load_file, save_file

    if destination.exists():
        raise ValueError(f"refuse to overwrite {destination}")
    configs = [json.loads((s / CONFIG).read_text()) for s in sources]
    tensors = [load_file(s / WEIGHTS) for s in sources]
    if any(t.dtype != torch.float32 for ts in tensors for t in ts.values()):
        raise ValueError("FP32 masters are required")
    config = averaged_config(configs)
    stacked = stack(tensors)
    destination.mkdir(parents=True)
    for path in sources[0].iterdir():
        if path.is_file() and path.name not in {WEIGHTS, CONFIG}:
            if path.name != "training_metadata.json":
                shutil.copy2(path, destination / path.name)
    (destination / CONFIG).write_text(json.dumps(config, indent=2) + "\n")
    save_file(stacked, destination / WEIGHTS, metadata={"format": "pt"})
    scale = configs[0]["lora_alpha"] / configs[0]["r"]
    worst = 0.0
    modules = sorted({k.split(".lora_")[0] for k in stacked})
    for module in modules:
        expected = sum(update(t, module, scale) for t in tensors) / len(tensors)
        actual = update(stacked, module, config["lora_alpha"] / config["r"])
        error = ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item()
        worst = max(worst, error)
    if worst > 1e-6:
        raise ValueError(f"stacked adapter is not the exact average: {worst}")
    return {"modules": len(modules), "max_relative_update_error": worst, **config}
