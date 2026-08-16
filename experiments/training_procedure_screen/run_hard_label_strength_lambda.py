#!/usr/bin/env python3
"""Queue and run the hard-label strength curve on two Lambda H100s."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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
)
from experiments.training_procedure_screen.core import (  # noqa: E402
    balanced_lanes,
    validate_hard_label_strength_jobs,
)
from experiments.training_procedure_screen.run_lambda import (  # noqa: E402
    run_evaluation_lane,
    run_preflight_lane,
)
from gleipnir.qwen35_fast_training import (  # noqa: E402
    DEFAULT_FLA_TARGET,
    ensure_fla_kernels,
)


def write_atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def wait_for_completion(
    dependency: Path, queue_status: Path, planned_jobs: list[str], interval: int
) -> None:
    """Wait for a prerequisite campaign while publishing visible queue state."""
    queued_at = time.time()
    while True:
        state = json.loads(dependency.read_text()) if dependency.exists() else {}
        dependency_state = state.get("state", "not_found")
        write_atomic_json(
            queue_status,
            {
                "state": "queued",
                "phase": "waiting_for_hard_anchor_replication",
                "queued_at_unix": queued_at,
                "updated_at_unix": time.time(),
                "waiting_for": dependency.as_posix(),
                "dependency_state": dependency_state,
                "planned_jobs": planned_jobs,
                "cells": len(planned_jobs),
            },
        )
        if dependency_state == "complete":
            return
        if dependency_state == "failed":
            raise RuntimeError("hard-anchor replication failed; refusing follow-on")
        time.sleep(interval)


def relocate_jobs(
    source: Path,
    destination: Path,
    student_rows: Path,
    soft_targets: Path,
    selection_manifest: Path,
    development: Path,
) -> list[dict[str, Any]]:
    root = destination.parent
    relocated = []
    for job in read_jsonl(source):
        output = root / "runs" / str(job["job_name"])
        selected = dict(job)
        selected.update(
            student_rows=student_rows.resolve().as_posix(),
            soft_targets=soft_targets.resolve().as_posix(),
            selection_manifest=selection_manifest.resolve().as_posix(),
            validation=development.resolve().as_posix(),
            output_dir=output.resolve().as_posix(),
            causal_adapter_dir=(output / "causal_adapter").resolve().as_posix(),
            model_dir=(output / "model").resolve().as_posix(),
        )
        relocated.append(selected)
    validate_hard_label_strength_jobs(relocated)
    write_jsonl(destination, relocated)
    return relocated


def preflight_job(jobs: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    """Create a two-step hard-only canary for the only new loss endpoint."""
    source = next(job for job in jobs if job["strength_id"] == "hard-only")
    output = path.parent / "preflight" / "hard-only"
    job = {
        **source,
        "job_name": "preflight-hard-only",
        "max_steps": 2,
        "save_strategy": "no",
        "output_dir": output.as_posix(),
        "causal_adapter_dir": (output / "causal_adapter").as_posix(),
        "model_dir": (output / "model").as_posix(),
    }
    write_jsonl(path, [job])
    return job


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("results/training_procedure_screen/hard_label_strength")
    parser.add_argument("--jobs", type=Path, default=root / "jobs.jsonl")
    parser.add_argument("--lambda-jobs", type=Path, default=root / "lambda_jobs.jsonl")
    parser.add_argument("--status", type=Path, default=root / "lambda_status.json")
    parser.add_argument(
        "--wait-for-status",
        type=Path,
        default=Path(
            "results/training_procedure_screen/hard_anchor_replication/"
            "lambda_status.json"
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
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
        default=Path("data/training_procedure_screen/train_full_tuning_pool.jsonl"),
    )
    parser.add_argument(
        "--development",
        type=Path,
        default=Path("data/optimization_regularization_screen/development.jsonl"),
    )
    parser.add_argument(
        "--screen-results",
        type=Path,
        default=Path("results/training_procedure_screen/summary/results.csv"),
    )
    parser.add_argument(
        "--replication-results",
        type=Path,
        default=Path(
            "results/training_procedure_screen/hard_anchor_replication/summary/"
            "replication_results.csv"
        ),
    )
    parser.add_argument("--fla-target", type=Path, default=DEFAULT_FLA_TARGET)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--revision", default=os.environ.get("GLEIPNIR_COMMIT"))
    args = parser.parse_args()
    if args.gpus != 2:
        raise ValueError("the frozen strength curve requires exactly two H100s")
    jobs_path = args.lambda_jobs.resolve()
    development = args.development.resolve()
    jobs = relocate_jobs(
        args.jobs.resolve(),
        jobs_path,
        args.student_rows.resolve(),
        args.soft_targets.resolve(),
        args.selection_manifest.resolve(),
        development,
    )
    status_path = args.status.resolve()
    try:
        wait_for_completion(
            args.wait_for_status.resolve(),
            status_path,
            [str(job["job_name"]) for job in jobs],
            args.poll_seconds,
        )
        status = Status(
            status_path,
            jobs,
            run_metadata={
                "revision": args.revision,
                "phase": "kernel_preflight",
                "cells": len(jobs),
                "training_rows": int(jobs[0]["train_rows"]),
                "development_rows": 1972,
                "seeds": [0, 1, 2],
                "new_strengths": 11,
                "competition_validation_used": False,
                "final_test_used": False,
            },
        )
        if args.revision is not None:
            os.environ["GLEIPNIR_COMMIT"] = args.revision
        os.environ.update(ensure_fla_kernels(args.fla_target))
        status.set_phase("hard_only_preflight")
        preflight_path = jobs_path.parent / "preflight_jobs.jsonl"
        canary = preflight_job(jobs, preflight_path)
        run_preflight_lane(0, [canary], preflight_path, development)

        status.set_phase("hard_label_strength_training")
        lanes = balanced_lanes(jobs, args.gpus)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(run_lora_lane, lane, lane_jobs, jobs_path, status)
                for lane, lane_jobs in enumerate(lanes)
            ]
            for future in futures:
                future.result()
        status.set_phase("causal_final_evaluation")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    run_evaluation_lane, lane, lane_jobs, jobs_path, development
                )
                for lane, lane_jobs in enumerate(lanes)
            ]
            for future in futures:
                future.result()
        status.set_phase("summarizing")
        subprocess.run(
            [
                sys.executable,
                "experiments/training_procedure_screen/summarize_hard_label_strength.py",
                "--jobs",
                jobs_path.as_posix(),
                "--screen-results",
                args.screen_results.resolve().as_posix(),
                "--replication-results",
                args.replication_results.resolve().as_posix(),
            ],
            cwd=ROOT,
            check=True,
        )
    except BaseException as error:
        if "status" in locals():
            status.fail(error)
        else:
            write_atomic_json(status_path, {"state": "failed", "error": repr(error)})
        raise
    status.finish()


if __name__ == "__main__":
    main()
