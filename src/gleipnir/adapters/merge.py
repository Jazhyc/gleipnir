"""Materialize standard FP32 LoRA updates into disposable BF16 serving weights."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    """Hash a model file without allocating its complete contents."""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def lora_pairs(keys: list[str], config: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Map standard PEFT serving keys onto checkpoint weights, rejecting extras."""
    if config.get("peft_type") != "LORA" or config.get("bias", "none") != "none":
        raise ValueError("merge requires a standard bias-free LoRA adapter")
    for flag in (
        "use_dora",
        "use_rslora",
        "use_qalora",
        "use_bdlora",
        "fan_in_fan_out",
    ):
        if config.get(flag):
            raise ValueError(f"unsupported merge variant: {flag}")
    for field in (
        "modules_to_save",
        "rank_pattern",
        "alpha_pattern",
        "target_parameters",
        "alora_invocation_tokens",
        "layer_replication",
        "trainable_token_indices",
    ):
        if config.get(field):
            raise ValueError(f"unsupported merge field: {field}")
    pairs = {}
    remaining = set(keys)
    for key in keys:
        if not key.startswith("base_model.model.") or not key.endswith(
            ".lora_A.weight"
        ):
            continue
        other = key.removesuffix(".lora_A.weight") + ".lora_B.weight"
        if other not in remaining:
            raise ValueError(f"missing LoRA B tensor: {key}")
        target = (
            key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
            + ".weight"
        )
        pairs[target] = (key, other)
        remaining.difference_update((key, other))
    if not pairs or remaining:
        raise ValueError(f"unrepresented adapter tensors: {sorted(remaining)[:3]}")
    return pairs


def merge_weight(weight: Any, a: Any, b: Any, *, rank: int, alpha: float) -> Any:
    """Accumulate a standard LoRA update in FP32 and round once to BF16."""
    import torch

    if (
        weight.dtype != torch.bfloat16
        or a.dtype != torch.float32
        or b.dtype != torch.float32
    ):
        raise ValueError("merge expects BF16 base weights and FP32 master adapters")
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
        raise ValueError("LoRA rank/shape mismatch")
    if tuple(weight.shape) != (b.shape[0], a.shape[1]):
        raise ValueError("base/adapter projection shape mismatch")
    if not all(bool(torch.isfinite(t).all()) for t in (weight, a, b)):
        raise ValueError("nonfinite merge input")
    merged = weight.float() + (b @ a) * (alpha / rank)
    if not bool(torch.isfinite(merged).all()):
        raise ValueError("nonfinite merged weight")
    rounded = merged.to(torch.bfloat16)
    if not bool(torch.isfinite(rounded).all()):
        raise ValueError("nonfinite BF16 serving weight")
    return rounded


def merge_checkpoint(
    base: Path,
    adapter: Path,
    destination: Path,
    *,
    expected_adapter_sha256: str,
    model_id: str,
    revision: str,
) -> dict:
    """Stream source shards, retaining all non-adapter tensors and model assets."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    weights = adapter / "adapter_model.safetensors"
    if file_sha256(weights) != expected_adapter_sha256:
        raise ValueError("merge adapter checksum drift")
    config = json.loads((adapter / "adapter_config.json").read_text())
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    if rank < 1 or alpha <= 0:
        raise ValueError("invalid LoRA scale")
    index_path = base / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    mapping = index["weight_map"]
    with safe_open(weights, framework="pt", device="cpu") as source:
        pairs = lora_pairs(list(source.keys()), config)
    if not set(pairs).issubset(mapping):
        raise ValueError(
            f"adapter/base key layout mismatch: {sorted(set(pairs) - set(mapping))[:3]}"
        )
    if destination.exists():
        raise FileExistsError("preserve existing merged artifact before replacement")
    destination.mkdir(parents=True)
    started = time.perf_counter()
    source_hashes, changed, merged_modules = {}, [], []
    with safe_open(weights, framework="pt", device="cpu") as source:
        for name in sorted(set(mapping.values())):
            shard = base / name
            source_hashes[name] = file_sha256(shard)
            tensors = {}
            with safe_open(shard, framework="pt", device="cpu") as reader:
                for key in reader.keys():
                    value = reader.get_tensor(key)
                    if key in pairs:
                        a, b = pairs[key]
                        merged = merge_weight(
                            value,
                            source.get_tensor(a),
                            source.get_tensor(b),
                            rank=rank,
                            alpha=alpha,
                        )
                        if not torch.equal(value, merged):
                            changed.append(key)
                        value = merged
                        merged_modules.append(key)
                    tensors[key] = value
                save_file(tensors, destination / name, metadata={"format": "pt"})
            print(
                f"merge_shard_complete={name} modules={len(merged_modules)}", flush=True
            )
    if set(merged_modules) != set(pairs) or not changed:
        raise ValueError("incomplete merge or zero adapter effect on weights")
    for path in base.iterdir():
        if path.is_file() and path.suffix in {
            ".json",
            ".txt",
            ".model",
            ".tiktoken",
            ".jinja",
        }:
            shutil.copy2(path, destination / path.name)
    files = {
        p.name: file_sha256(p) for p in sorted(destination.iterdir()) if p.is_file()
    }
    manifest = {
        "model": model_id,
        "revision": revision,
        "adapter": str(adapter),
        "adapter_sha256": expected_adapter_sha256,
        "adapter_config_sha256": file_sha256(adapter / "adapter_config.json"),
        "source_files_sha256": {
            **source_hashes,
            "config.json": file_sha256(base / "config.json"),
        },
        "method": "FP32 W + (alpha/r) B@A, finite checks, one BF16 serving cast",
        "master_preserved": True,
        "base_preserved": True,
        "merged_projection_count": len(merged_modules),
        "changed_projection_count": len(changed),
        "destination": str(destination),
        "files_sha256": files,
        "seconds": time.perf_counter() - started,
        "persistent_source_only": True,
    }
    (destination / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest
