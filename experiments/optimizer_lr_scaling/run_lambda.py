#!/usr/bin/env python3
"""Run the paired AdamW/Muon learning-rate screen on two Lambda H100s."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapter_capacity_scaling.prepare import (  # noqa: E402
    read_jsonl,
    write_jsonl,
)
from experiments.adapter_capacity_scaling.run_lambda import (  # noqa: E402
    Status,
    run_lora_lane,
    runtime_environment,
)
from experiments.optimizer_lr_scaling.core import (  # noqa: E402
    paired_optimizer_lanes,
)
from gleipnir.qwen35_fast_training import (  # noqa: E402
    DEFAULT_FLA_TARGET,
    ensure_fla_kernels,
)


def relocate_jobs(
    source: Path,
    destination: Path,
    student_rows: Path,
    soft_targets: Path,
    selection_manifest: Path,
    development: Path,
) -> list[dict[str, Any]]:
    """Rewrite machine-local paths while preserving optimizer job identity."""
    sweep_root = destination.parent
    relocated = []
    for job in read_jsonl(source):
        job_name = str(job["job_name"])
        output_dir = sweep_root / "runs" / job_name
        selected = dict(job)
        selected.update(
            {
                "student_rows": student_rows.resolve().as_posix(),
                "soft_targets": soft_targets.resolve().as_posix(),
                "selection_manifest": selection_manifest.resolve().as_posix(),
                "validation": development.resolve().as_posix(),
                "output_dir": output_dir.resolve().as_posix(),
                "causal_adapter_dir": (
                    output_dir / "causal_adapter"
                ).resolve().as_posix(),
                "model_dir": (output_dir / "model").resolve().as_posix(),
            }
        )
        relocated.append(selected)
    write_jsonl(destination, relocated)
    return relocated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("results/optimizer_lr_scaling/jobs.jsonl"),
    )
    parser.add_argument(
        "--lambda-jobs",
        type=Path,
        default=Path("results/optimizer_lr_scaling/lambda_jobs.jsonl"),
    )
    parser.add_argument(
        "--student-rows",
        type=Path,
        default=Path("data/data_scaling/student_rows.jsonl"),
    )
    parser.add_argument(
        "--soft-targets",
        type=Path,
        default=Path("data/data_scaling/soft_targets.jsonl"),
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path("data/optimizer_lr_scaling/train_seed0_f025.jsonl"),
    )
    parser.add_argument(
        "--development",
        type=Path,
        default=Path("data/optimizer_lr_scaling/development.jsonl"),
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=Path("results/optimizer_lr_scaling/lambda_status.json"),
    )
    parser.add_argument("--fla-target", type=Path, default=DEFAULT_FLA_TARGET)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--revision", default=os.environ.get("GLEIPNIR_COMMIT"))
    args = parser.parse_args()
    if args.gpus != 2:
        raise ValueError("the paired optimizer screen requires exactly two H100s")

    jobs_path = args.lambda_jobs.resolve()
    jobs = relocate_jobs(
        args.jobs.resolve(),
        jobs_path,
        args.student_rows.resolve(),
        args.soft_targets.resolve(),
        args.selection_manifest.resolve(),
        args.development.resolve(),
    )
    if len(jobs) != 10:
        raise ValueError(f"expected ten paired optimizer jobs, got {len(jobs)}")
    lanes = paired_optimizer_lanes(jobs)
    status = Status(
        args.status.resolve(),
        jobs,
        run_metadata={
            "revision": args.revision,
            "phase": "optimizer_training",
            "training_recipe": "qlora-nf4-double-quant-bf16-fla-0.5.2",
            "rank": 64,
            "optimizers": ["adamw", "muon"],
            "learning_rates": [1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
            "muon_adjust_lr_fn": "match_rms_adamw",
            "lr_scheduler_type": "linear",
            "warmup_ratio": 0.03,
            "internal_development": args.development.resolve().as_posix(),
            "competition_validation_used_for_selection": False,
        },
    )
    try:
        if args.revision is not None:
            os.environ["GLEIPNIR_COMMIT"] = args.revision
        status.set_phase("kernel_preflight")
        os.environ.update(ensure_fla_kernels(args.fla_target))
        status.set_phase("optimizer_training")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(run_lora_lane, index, lane, jobs_path, status)
                for index, lane in enumerate(lanes)
            ]
            for future in futures:
                future.result()

        status.set_phase("internal_development_evaluation")
        subprocess.run(
            [
                sys.executable,
                "experiments/adapter_capacity_scaling/evaluate.py",
                "--jobs",
                jobs_path.as_posix(),
                "--validation",
                args.development.resolve().as_posix(),
                "--mode",
                "lora",
            ],
            cwd=ROOT,
            env=runtime_environment("0"),
            check=True,
        )
        status.set_phase("summarizing")
        subprocess.run(
            [
                sys.executable,
                "experiments/optimizer_lr_scaling/summarize.py",
                "--jobs",
                jobs_path.as_posix(),
            ],
            cwd=ROOT,
            check=True,
        )
    except BaseException as error:
        status.fail(error)
        raise
    status.finish()


if __name__ == "__main__":
    main()
