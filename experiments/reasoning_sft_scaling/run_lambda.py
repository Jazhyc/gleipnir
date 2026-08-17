#!/usr/bin/env python3
"""Run the resumable reasoning-SFT evaluation on one Lambda GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write_status(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument(
        "--jobs", type=Path, default=Path("results/reasoning_sft_scaling/jobs.jsonl")
    )
    args = parser.parse_args()
    status = args.jobs.resolve().parent / "lambda_status.json"
    started = time.time()
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
    )
    write_status(status, state="evaluating", gpu=args.gpu, started_at_unix=started)
    try:
        subprocess.run(
            [
                sys.executable,
                "experiments/reasoning_sft_scaling/evaluate.py",
                "--jobs",
                args.jobs.resolve().as_posix(),
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "experiments/reasoning_sft_scaling/summarize.py",
                "--jobs",
                args.jobs.resolve().as_posix(),
            ],
            cwd=ROOT,
            check=True,
        )
    except BaseException as error:
        write_status(
            status,
            state="failed",
            gpu=args.gpu,
            started_at_unix=started,
            failed_at_unix=time.time(),
            error=repr(error),
        )
        raise
    write_status(
        status,
        state="complete",
        gpu=args.gpu,
        started_at_unix=started,
        completed_at_unix=time.time(),
    )


if __name__ == "__main__":
    main()
