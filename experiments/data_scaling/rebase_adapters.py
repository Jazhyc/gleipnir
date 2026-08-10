#!/usr/bin/env python3
"""Rebase text-only Qwen3.5 LoRAs onto its multimodal language-model path."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data_scaling.prepare import read_jsonl  # noqa: E402, I001


SOURCE_PREFIX = "base_model.model.model."
DESTINATION_PREFIX = "base_model.model.model.language_model."
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
    if not key.startswith(SOURCE_PREFIX):
        raise ValueError(f"unexpected causal adapter key: {key}")
    return DESTINATION_PREFIX + key.removeprefix(SOURCE_PREFIX)


def rebase_adapter(source_dir: Path, destination_dir: Path) -> dict[str, Any]:
    """Create a non-destructive, checksum-tracked rebased adapter copy."""
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
            and manifest.get("destination_sha256") == sha256_file(destination_weights)
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
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs", type=Path, default=Path("results/data_scaling/lambda_jobs.jsonl")
    )
    parser.add_argument(
        "--output-jobs",
        type=Path,
        default=Path("results/data_scaling/lambda_jobs_rebased.jsonl"),
    )
    parser.add_argument("--directory-name", default="adapter_image_text_to_text")
    args = parser.parse_args()

    jobs = read_jsonl(args.jobs.resolve())
    rebased_jobs = []
    for job in jobs:
        source_dir = Path(job["adapter_dir"])
        destination_dir = source_dir.parent / args.directory_name
        manifest = rebase_adapter(source_dir, destination_dir)
        rebased = dict(job)
        rebased.update(
            {
                "source_adapter_dir": source_dir.resolve().as_posix(),
                "adapter_dir": destination_dir.resolve().as_posix(),
                "adapter_rebase_manifest": (
                    destination_dir / MANIFEST_NAME
                ).resolve().as_posix(),
                "adapter_rebase_source_sha256": manifest["source_sha256"],
                "adapter_rebase_destination_sha256": manifest[
                    "destination_sha256"
                ],
            }
        )
        rebased_jobs.append(rebased)
        print(f"rebased {job['job_name']} -> {destination_dir}", flush=True)
    write_jsonl(args.output_jobs.resolve(), rebased_jobs)


if __name__ == "__main__":
    main()
