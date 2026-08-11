#!/usr/bin/env python3
"""Preflight the largest QLoRA rank with the production FLA recipe."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapter_capacity_scaling.run_train import (  # noqa: E402
    training_command,
)
from gleipnir.qwen35_fast_training import (  # noqa: E402
    DEFAULT_FLA_TARGET,
    FLA_VERSION,
    ensure_fla_kernels,
)


def benchmark_job(output_dir: Path, rank: int, seed: int) -> dict[str, Any]:
    """Return the production-shaped synthetic job used by the preflight."""
    return {
        "job_name": f"preflight-seed{seed}-r{rank:03d}",
        "capacity_kind": "lora",
        "seed": seed,
        "rank": rank,
        "lora_alpha": 2 * rank,
        "student_rows": (ROOT / "data/data_scaling/student_rows.jsonl").as_posix(),
        "soft_targets": (ROOT / "data/data_scaling/soft_targets.jsonl").as_posix(),
        "output_dir": output_dir.as_posix(),
        "causal_adapter_dir": (output_dir / "causal_adapter").as_posix(),
        "model_dir": (output_dir / "model").as_posix(),
        "micro_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 32,
    }


def validate_preflight(
    metadata: dict[str, Any],
    *,
    minimum_samples_per_second: float,
) -> None:
    """Fail closed on a fallback kernel, nonstandard QLoRA, or slow result."""
    expected_quantization = {
        "enabled": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    }
    if metadata.get("quantization") != expected_quantization:
        raise RuntimeError(f"unexpected QLoRA metadata: {metadata.get('quantization')}")
    fla = metadata.get("flash_linear_attention")
    if not isinstance(fla, dict) or fla.get("version") != FLA_VERSION:
        raise RuntimeError(f"FLA preflight failed: {fla}")
    if fla.get("available") is not True or fla.get("required") is not True:
        raise RuntimeError(f"FLA fallback detected: {fla}")
    if metadata.get("training_batch") != {
        "micro_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 32,
    }:
        raise RuntimeError(
            f"unexpected training batch: {metadata.get('training_batch')}"
        )
    if metadata.get("direct_logits_mode") != "selected_positions":
        raise RuntimeError(
            f"unexpected logit mode: {metadata.get('direct_logits_mode')}"
        )
    kernel_modules = metadata.get("gated_delta_kernel_modules")
    if not isinstance(kernel_modules, list) or not kernel_modules:
        raise RuntimeError(f"missing gated-delta kernel metadata: {kernel_modules}")
    if any(not module.startswith("fla.ops.") for module in kernel_modules):
        raise RuntimeError(f"Torch gated-delta fallback detected: {kernel_modules}")
    throughput = float(metadata["train_metrics"]["train_samples_per_second"])
    if throughput < minimum_samples_per_second:
        raise RuntimeError(
            f"QLoRA preflight throughput {throughput:.3f} is below "
            f"{minimum_samples_per_second:.3f} samples/s"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--minimum-samples-per-second", type=float, default=4.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/adapter_capacity_scaling_qlora/preflight/seed0-r256-b8"
        ),
    )
    parser.add_argument("--fla-target", type=Path, default=DEFAULT_FLA_TARGET)
    args = parser.parse_args()
    if args.rank != 256 or args.max_steps < 20:
        raise ValueError("production preflight requires rank 256 and at least 20 steps")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = ensure_fla_kernels(args.fla_target)
    job = benchmark_job(output_dir, args.rank, args.seed)
    command = training_command(job, distributed_processes=1)
    command.extend(
        [
            f"student.training.max_steps={args.max_steps}",
            f"student.output_dir={job['causal_adapter_dir']}",
        ]
    )
    started_at = time.time()
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    metadata_path = Path(job["causal_adapter_dir"]) / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    validate_preflight(
        metadata,
        minimum_samples_per_second=args.minimum_samples_per_second,
    )
    summary = {
        "state": "complete",
        "started_at_unix": started_at,
        "completed_at_unix": time.time(),
        "commit": os.environ.get("GLEIPNIR_COMMIT"),
        "rank": args.rank,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "minimum_samples_per_second": args.minimum_samples_per_second,
        "metadata": metadata,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"QLoRA rank-{args.rank} preflight passed: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
