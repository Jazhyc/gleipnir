"""Run the frozen serial training and persistent-vLLM ID campaign, fail closed."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys

from experiments.monitoring_hard_labels.prepare import (
    FRACTIONS,
    OUTPUT,
    ROOT,
    verify_preparation,
    write_json,
)
from gleipnir.monitoring_campaign_runtime import training_environment


def run(*, resume_evaluation: bool = False) -> None:
    verify_preparation()
    if resume_evaluation:
        for variant in FRACTIONS:
            directory = OUTPUT / "4b" / variant
            for name in ("complete.json", "parity_reference.json"):
                if not (directory / name).is_file():
                    raise ValueError(f"evaluation resume requires {variant}/{name}")
    logs = ROOT / "logs/runpod/monitoring_hard_labels"
    logs.mkdir(parents=True, exist_ok=True)
    serving_env = dict(os.environ)
    serving_env.update(
        PATH=os.pathsep.join(
            [
                str(ROOT / ".venv/bin"),
                f"{os.environ.get('CUDA_HOME', '/usr/local/cuda')}/bin",
                os.environ.get("PATH", ""),
            ]
        ),
        VLLM_CACHE_ROOT=str(ROOT / ".cache/vllm/monitoring_hard_labels_v1"),
        TORCHINDUCTOR_CACHE_DIR=str(
            ROOT / ".cache/torchinductor/monitoring_hard_labels_v1"
        ),
        PYTHONUNBUFFERED="1",
        WANDB_MODE="disabled",
    )
    stages = [
        (
            variant,
            "experiments.monitoring_hard_labels.train",
            ["--variant", variant],
            serving_env,
        )
        for variant in FRACTIONS
    ]
    stages.extend(
        [
            (
                "reference",
                "experiments.monitoring_hard_labels.evaluate",
                ["--backend", "reference"],
                training_environment(ROOT, "monitoring_hard_labels_v1"),
            ),
            (
                "vllm",
                "experiments.monitoring_hard_labels.evaluate",
                ["--backend", "vllm"],
                serving_env,
            ),
            (
                "summary",
                "experiments.monitoring_hard_labels.summarize",
                [],
                serving_env,
            ),
        ]
    )
    if resume_evaluation:
        stages = stages[-2:]
    for stage, module, args, env in stages:
        write_json(OUTPUT / "status.json", {"stage": stage, "status": "running"})
        print(f"campaign_stage {stage}", flush=True)
        with (logs / f"{stage}.log").open("a") as log:
            try:
                subprocess.run(
                    [sys.executable, "-m", module, *args],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            except subprocess.CalledProcessError as error:
                write_json(
                    OUTPUT / "status.json",
                    {
                        "stage": stage,
                        "status": "failed",
                        "returncode": error.returncode,
                    },
                )
                raise
    write_json(OUTPUT / "status.json", {"status": "complete"})
    print("campaign_complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume-evaluation",
        action="store_true",
        help="Resume serving and summary using completed training/reference receipts",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "campaign.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(resume_evaluation=args.resume_evaluation)


if __name__ == "__main__":
    main()
