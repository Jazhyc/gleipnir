#!/usr/bin/env python3
"""Merge the benchmark LoRA into pinned BF16 base weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.classifier_throughput.core import sha256_file  # noqa: E402


def tree_manifest(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "merge_manifest.json"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    adapter = Path(config["adapter"]).resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "merge_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        observed = tree_manifest(output_dir)
        if observed == manifest.get("files"):
            print(f"reusing verified merged model at {output_dir}", flush=True)
            return
        raise RuntimeError(f"merged-model manifest mismatch at {output_dir}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"loading pinned BF16 base model for merge: {config['model']}", flush=True)
    base = AutoModelForImageTextToText.from_pretrained(
        config["model"],
        revision=config["model_revision"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    adapted = PeftModel.from_pretrained(base, adapter.as_posix())
    print(f"merging adapter: {adapter}", flush=True)
    merged = adapted.merge_and_unload(safe_merge=True)
    merged.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    processor = AutoProcessor.from_pretrained(
        config["model"],
        revision=config["model_revision"],
    )
    processor.save_pretrained(output_dir)
    files = tree_manifest(output_dir)
    manifest = {
        "manifest_version": 2,
        "model": config["model"],
        "model_revision": config["model_revision"],
        "adapter": adapter.as_posix(),
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "config_sha256": sha256_file(config_path),
        "files": files,
        "tree_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        "dtype": "bfloat16",
        "safe_merge": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"merged model written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
