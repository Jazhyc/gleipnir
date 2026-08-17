#!/usr/bin/env python3
"""Rebase the immutable AQ reasoning-SFT fraction adapters for vLLM serving."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reasoning_sft_scaling.core import (  # noqa: E402
    CANONICAL_PREFIX,
    VISION_EXCLUDE_PATTERN,
)
from gleipnir.qwen35_adapter_rebase import rebase_adapter  # noqa: E402

DEFAULT_AQ_ROOT = Path("/scratch/s4626451/Aletheias-Quest-Competition")
ADAPTERS = (
    (0.01, 36, "qwen9b_pid_varied_datafrac01_adamw5e5_v1"),
    (0.02, 54, "qwen9b_pid_varied_datafrac02_adamw5e5_v1"),
    (0.05, 144, "qwen9b_pid_varied_datafrac05_adamw5e5_v1"),
    (0.10, 288, "qwen9b_pid_varied_datafrac10_adamw5e5_v1"),
    (0.20, 576, "qwen9b_pid_varied_datafrac20_adamw5e5_v1"),
    (0.40, 1152, "qwen9b_pid_varied_datafrac40_adamw5e5_v1"),
    (0.80, 2301, "qwen9b_pid_varied_datafrac80_adamw5e5_v1"),
    (1.00, 2877, "qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_and_patch_config(adapter_dir: Path) -> None:
    config_path = adapter_dir / "adapter_config.json"
    config = json.loads(config_path.read_text())
    if int(config["r"]) != 16 or int(config["lora_alpha"]) != 32:
        raise ValueError(f"unexpected LoRA recipe in {config_path}")
    config["exclude_modules"] = VISION_EXCLUDE_PATTERN
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def first_tensor_key(path: Path) -> tuple[str, int]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    if not keys:
        raise ValueError(f"adapter contains no tensors: {path}")
    return keys[0], len(keys)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aq-root", type=Path, default=DEFAULT_AQ_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/reasoning_sft_scaling")
    )
    args = parser.parse_args()

    aq_root = args.aq_root.resolve()
    output_dir = args.output_dir.resolve()
    jobs = []
    for fraction, train_rows, source_name in ADAPTERS:
        source_run = aq_root / "results" / "blackbox" / source_name
        source_adapter = source_run / "adapter"
        job_name = f"f{round(fraction * 100):03d}"
        destination = output_dir / "adapters" / job_name
        manifest = rebase_adapter(source_adapter, destination)
        validate_and_patch_config(destination)
        shutil.copy2(source_run / "config.yaml", destination / "source_config.yaml")
        first_key, tensor_count = first_tensor_key(
            destination / "adapter_model.safetensors"
        )
        if not first_key.startswith(CANONICAL_PREFIX) or tensor_count != 256:
            raise ValueError(
                f"invalid rebased layout for {job_name}: {tensor_count=} {first_key=}"
            )
        jobs.append(
            {
                "job_name": job_name,
                "fraction": fraction,
                "train_rows": train_rows,
                "seed": 0,
                "source_run": source_run.as_posix(),
                "source_adapter": source_adapter.as_posix(),
                "adapter_dir": destination.relative_to(ROOT).as_posix(),
                "source_weights_sha256": manifest["source_sha256"],
                "serving_weights_sha256": manifest["destination_sha256"],
                "serving_config_sha256": sha256_file(
                    destination / "adapter_config.json"
                ),
                "tensor_count": tensor_count,
            }
        )
        print(f"prepared {job_name}: {train_rows} rows -> {destination}", flush=True)
    write_jsonl(output_dir / "jobs.jsonl", jobs)


if __name__ == "__main__":
    main()
