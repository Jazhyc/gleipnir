#!/usr/bin/env python3
"""Run the missing data-volume by QLoRA-rank cells on two Lambda H100s."""

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
from experiments.data_capacity_scaling.core import (  # noqa: E402
    balanced_interaction_lanes,
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
    validation: Path,
) -> list[dict[str, Any]]:
    """Relocate new jobs and their selection manifests to the remote root."""
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
                "validation": validation.resolve().as_posix(),
                "selection_manifest": (
                    sweep_root / "selections" / Path(job["selection_manifest"]).name
                ).resolve().as_posix(),
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
        default=Path("results/data_capacity_scaling_qlora/jobs.jsonl"),
    )
    parser.add_argument(
        "--lambda-jobs",
        type=Path,
        default=Path("results/data_capacity_scaling_qlora/lambda_jobs.jsonl"),
    )
    parser.add_argument(
        "--full-data-jobs",
        type=Path,
        default=Path(
            "results/adapter_capacity_scaling_qlora/lambda_jobs.jsonl"
        ),
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
        "--validation",
        type=Path,
        default=Path("data/data_scaling/validation.jsonl"),
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=Path("results/data_capacity_scaling_qlora/lambda_status.json"),
    )
    parser.add_argument("--fla-target", type=Path, default=DEFAULT_FLA_TARGET)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--revision", default=os.environ.get("GLEIPNIR_COMMIT"))
    args = parser.parse_args()
    if args.gpus != 2:
        raise ValueError("this schedule is predeclared for exactly two H100 GPUs")

    jobs_path = args.lambda_jobs.resolve()
    jobs = relocate_jobs(
        args.jobs.resolve(),
        jobs_path,
        args.student_rows.resolve(),
        args.soft_targets.resolve(),
        args.validation.resolve(),
    )
    if len(jobs) != 36:
        raise ValueError(f"expected 36 new interaction jobs, got {len(jobs)}")
    status = Status(
        args.status.resolve(),
        jobs,
        run_metadata={
            "revision": args.revision,
            "training_recipe": "qlora-nf4-double-quant-bf16-fla-0.5.2",
            "direct_logits_mode": "selected_positions",
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 32,
            "full_data_reference_jobs": args.full_data_jobs.resolve().as_posix(),
        },
    )
    try:
        if args.revision is not None:
            os.environ["GLEIPNIR_COMMIT"] = args.revision
        status.set_phase("kernel_preflight")
        os.environ.update(ensure_fla_kernels(args.fla_target))
        status.set_phase("interaction_training")
        lanes = balanced_interaction_lanes(jobs, args.gpus)
        with ThreadPoolExecutor(max_workers=args.gpus) as executor:
            futures = [
                executor.submit(run_lora_lane, index, lane, jobs_path, status)
                for index, lane in enumerate(lanes)
            ]
            for future in futures:
                future.result()

        status.set_phase("interaction_evaluation")
        subprocess.run(
            [
                sys.executable,
                "experiments/adapter_capacity_scaling/evaluate.py",
                "--jobs",
                jobs_path.as_posix(),
                "--validation",
                args.validation.resolve().as_posix(),
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
                "experiments/data_capacity_scaling/summarize.py",
                "--jobs",
                jobs_path.as_posix(),
                "--full-data-jobs",
                args.full_data_jobs.resolve().as_posix(),
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
