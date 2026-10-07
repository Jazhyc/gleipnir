"""Checksum-tracked Qwen3.5 causal-to-multimodal LoRA export."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

SOURCE_PREFIX = "base_model.model.model."
DESTINATION_PREFIX = "base_model.model.model.language_model."
PASSTHROUGH_PREFIXES = (
    "base_model.model.lm_head.",
    "base_model.model.decision_head.",
)
WEIGHTS_NAME = "adapter_model.safetensors"
MANIFEST_NAME = "rebase_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebase_key(key: str) -> str:
    """Map a causal-wrapper PEFT key to the Qwen3.5 multimodal wrapper."""
    if key.startswith(DESTINATION_PREFIX):
        raise ValueError(f"adapter key is already rebased: {key}")
    if key.startswith(PASSTHROUGH_PREFIXES):
        return key
    if not key.startswith(SOURCE_PREFIX):
        raise ValueError(f"unexpected causal adapter key: {key}")
    return DESTINATION_PREFIX + key.removeprefix(SOURCE_PREFIX)


def rebase_adapter(source_dir: Path, destination_dir: Path) -> dict[str, Any]:
    """Create a non-destructive, checksum-tracked rebased adapter copy."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    source_dir = source_dir.resolve()
    destination_dir = destination_dir.resolve()
    source_weights = source_dir / WEIGHTS_NAME
    source_config = source_dir / "adapter_config.json"
    if not source_weights.is_file() or not source_config.is_file():
        raise FileNotFoundError(f"incomplete source adapter: {source_dir}")
    if source_dir == destination_dir:
        raise ValueError("source and destination adapter directories must differ")

    source_sha256 = sha256_file(source_weights)
    manifest_path = destination_dir / MANIFEST_NAME
    destination_weights = destination_dir / WEIGHTS_NAME
    if manifest_path.is_file() and destination_weights.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("source_sha256") == source_sha256
            and manifest.get("destination_sha256")
            == sha256_file(destination_weights)
            and manifest.get("source_prefix") == SOURCE_PREFIX
            and manifest.get("destination_prefix") == DESTINATION_PREFIX
        ):
            return manifest
        raise ValueError(f"stale or modified rebased adapter: {destination_dir}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.is_file() and path.name not in {WEIGHTS_NAME, MANIFEST_NAME}:
            shutil.copy2(path, destination_dir / path.name)

    tensors = {}
    metadata = None
    with safe_open(source_weights, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            rebased_key = rebase_key(key)
            if rebased_key in tensors:
                raise ValueError(f"rebased adapter key collision: {rebased_key}")
            tensors[rebased_key] = handle.get_tensor(key)
    if not tensors:
        raise ValueError(f"adapter contains no tensors: {source_weights}")

    temporary = destination_weights.with_suffix(".safetensors.tmp")
    save_file(tensors, temporary, metadata=metadata)
    temporary.replace(destination_weights)
    destination_sha256 = sha256_file(destination_weights)
    manifest = {
        "format": "gleipnir-qwen35-causal-to-image-text-to-text-lora-v1",
        "source_adapter": source_dir.as_posix(),
        "destination_adapter": destination_dir.as_posix(),
        "source_weights": source_weights.name,
        "source_sha256": source_sha256,
        "destination_sha256": destination_sha256,
        "tensor_count": len(tensors),
        "source_prefix": SOURCE_PREFIX,
        "destination_prefix": DESTINATION_PREFIX,
        "passthrough_prefixes": list(PASSTHROUGH_PREFIXES),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    """Rebase one causal adapter for serving without retraining it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            rebase_adapter(args.source, args.destination), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
