#!/usr/bin/env python3
"""Parity-check and run frozen external transfer with vLLM on two H100s."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapter_capacity_scaling.run_lambda import (  # noqa: E402
    Status,
    runtime_environment,
)
from gleipnir.qwen35_fast_training import (  # noqa: E402
    DEFAULT_FLA_TARGET,
    ensure_fla_kernels,
)


def vllm_evaluation_command(
    evaluation: Path,
    manifest: Path,
    jobs: Path,
    output_dir: Path,
    *,
    include_base: bool,
    strength_id: str,
) -> list[str]:
    command = [
        sys.executable,
        "experiments/aisi_lie_detection_transfer/evaluate_vllm.py",
        "--evaluation",
        evaluation.as_posix(),
        "--manifest",
        manifest.as_posix(),
        "--jobs",
        jobs.as_posix(),
        "--output-dir",
        output_dir.as_posix(),
        "--strength-id",
        strength_id,
    ]
    if include_base:
        command.append("--include-base")
    return command


def parity_commands(
    evaluation: Path,
    manifest: Path,
    jobs: Path,
    output_dir: Path,
    smoke_rows: int,
) -> list[list[str]]:
    """Return matched eager, vLLM, and comparison parity commands."""
    shared = [
        "--evaluation",
        evaluation.as_posix(),
        "--manifest",
        manifest.as_posix(),
        "--jobs",
        jobs.as_posix(),
        "--include-base",
        "--strength-id",
        "0500",
        "--seed",
        "0",
        "--smoke-rows",
        str(smoke_rows),
    ]
    eager_root = output_dir / "eager"
    vllm_root = output_dir / "vllm"
    return [
        [
            sys.executable,
            "experiments/aisi_lie_detection_transfer/evaluate_causal.py",
            *shared,
            "--output-dir",
            eager_root.as_posix(),
        ],
        [
            sys.executable,
            "experiments/aisi_lie_detection_transfer/evaluate_vllm.py",
            *shared,
            "--output-dir",
            vllm_root.as_posix(),
        ],
        [
            sys.executable,
            "experiments/aisi_lie_detection_transfer/compare_parity.py",
            "--eager-root",
            eager_root.as_posix(),
            "--vllm-root",
            vllm_root.as_posix(),
            "--output",
            (output_dir / "report.json").as_posix(),
        ],
    ]


def run_lane(
    lane: int,
    name: str,
    command: list[str],
    status: Status,
) -> None:
    status.start_job(name)
    subprocess.run(
        command,
        cwd=ROOT,
        env=runtime_environment(str(lane)),
        check=True,
    )
    status.finish_job(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("data/aisi_lie_detection_transfer/evaluation.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/aisi_lie_detection_transfer/manifest.json"),
    )
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path(
            "results/training_procedure_screen/hard_label_strength/jobs.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aisi_lie_detection_transfer"),
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=Path("results/aisi_lie_detection_transfer/status.json"),
    )
    parser.add_argument("--fla-target", type=Path, default=DEFAULT_FLA_TARGET)
    parser.add_argument("--parity-rows", type=int, default=64)
    parser.add_argument("--revision", default=os.environ.get("GLEIPNIR_COMMIT"))
    args = parser.parse_args()

    lanes = [
        {"job_name": "base-and-hard-only", "gpu": 0},
        {"job_name": "mixed-weight-0500", "gpu": 1},
    ]
    planned = [{"job_name": "backend-parity", "gpu": 0}, *lanes]
    status = Status(
        args.status.resolve(),
        planned,
        run_metadata={
            "state": "evaluating",
            "revision": args.revision,
            "phase": "preparing_external_evaluation",
            "gpus": 2,
            "serving_backend": "vllm_continuous_batching",
            "targets": ["base", "hard-only-seeds-0-1-2", "0500-seeds-0-1-2"],
            "selection_frozen": True,
        },
    )
    try:
        prepare = [
            sys.executable,
            "experiments/aisi_lie_detection_transfer/prepare.py",
            "--output",
            args.evaluation.resolve().as_posix(),
            "--manifest",
            args.manifest.resolve().as_posix(),
            "--revision",
            args.revision or "unknown",
        ]
        subprocess.run(prepare, cwd=ROOT, check=True)
        if not args.jobs.resolve().is_file():
            raise FileNotFoundError(f"missing hard-label job manifest: {args.jobs}")
        status.set_phase("backend_parity_eager")
        os.environ.update(ensure_fla_kernels(args.fla_target))
        status.start_job("backend-parity")
        parity = parity_commands(
            args.evaluation.resolve(),
            args.manifest.resolve(),
            args.jobs.resolve(),
            args.output_dir.resolve() / "parity",
            args.parity_rows,
        )
        subprocess.run(
            parity[0], cwd=ROOT, env=runtime_environment("0"), check=True
        )
        status.set_phase("backend_parity_vllm")
        subprocess.run(
            parity[1], cwd=ROOT, env=runtime_environment("0"), check=True
        )
        status.set_phase("backend_parity_comparison")
        subprocess.run(parity[2], cwd=ROOT, check=True)
        status.finish_job("backend-parity")
        status.set_phase("external_vllm_evaluation")
        runs_dir = args.output_dir.resolve() / "vllm_runs"
        commands = [
            vllm_evaluation_command(
                args.evaluation.resolve(),
                args.manifest.resolve(),
                args.jobs.resolve(),
                runs_dir,
                include_base=True,
                strength_id="hard-only",
            ),
            vllm_evaluation_command(
                args.evaluation.resolve(),
                args.manifest.resolve(),
                args.jobs.resolve(),
                runs_dir,
                include_base=False,
                strength_id="0500",
            ),
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    run_lane,
                    lane["gpu"],
                    lane["job_name"],
                    command,
                    status,
                )
                for lane, command in zip(lanes, commands, strict=True)
            ]
            for future in futures:
                future.result()
        status.set_phase("summarizing")
        subprocess.run(
            [
                sys.executable,
                "experiments/aisi_lie_detection_transfer/summarize.py",
                "--runs",
                runs_dir.as_posix(),
                "--output-dir",
                (args.output_dir.resolve() / "summary").as_posix(),
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
