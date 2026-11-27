"""Benchmark the selected scorer on one full workload across client concurrency."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

from experiments.b200_inference_benchmark.run import ROOT, write
from experiments.b200_inference_benchmark.run import environment as base_environment
from experiments.b200_monitor_score.run import SERVING, measure
from gleipnir.serving.score_runtime import resume_score_environment

EXPERIMENT = Path(__file__).parent
CONCURRENCIES = (1, 2, 4, 16, 32, 64, 128)


def validate_settings(settings: dict) -> None:
    """Freeze the user-requested sweep and one shared quality cohort."""
    expected = [
        {"concurrency": c, "workload": "full", "repeats": 3} for c in CONCURRENCIES
    ]
    if (
        settings["benchmark_passes"] != expected
        or settings["comparison_concurrency"] != 128
        or settings["warmup_repeats"] != 1
        or settings["promote"]
    ):
        raise ValueError("reference scaling contract changed")


def reference_command(command: list[str], settings: dict, hashes: dict) -> list[str]:
    """Require stock FCFS and the selected token/sequence/precision limits."""
    validate_settings(settings)
    if "--scheduler-cls" in command:
        raise ValueError("scaling requires the stock reference scheduler")
    for flag, value in (
        ("--runner", "pooling"),
        ("--max-num-seqs", "128"),
        ("--max-num-batched-tokens", "32768"),
    ):
        if command[command.index(flag) + 1] != value:
            raise ValueError("selected reference serving limits changed")
    return command.copy()


async def run(name: str, parent_path: Path, bootstrap: Path | None) -> None:
    parent = json.loads(parent_path.read_text())
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if apps or (SERVING / "server.json").exists():
        raise ValueError("scaling launch requires an idle GPU and no live scorer")
    env = base_environment()
    if bootstrap is not None:
        if not (bootstrap / "sitecustomize.py").is_file():
            raise ValueError("source bootstrap missing")
        env["PYTHONPATH"] = f"{bootstrap.resolve()}:{env['PYTHONPATH']}"
    env.pop("GLEIPNIR_LENGTH_ADMISSION_CONFIG", None)
    env = resume_score_environment(ROOT, parent, env)
    try:
        await measure(
            name,
            experiment=EXPERIMENT,
            result_group="b200_score_scaling",
            retired_parent=parent_path,
            resumed_environment=env,
            command_prepare=reference_command,
        )
    except BaseException as error:
        out = ROOT / "results/b200_score_scaling" / name
        if out.exists():
            write(out / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--retired-parent", type=Path, required=True)
    parser.add_argument("--source-bootstrap-dir", type=Path)
    args = parser.parse_args()
    if args.name in {"", ".", ".."} or Path(args.name).name != args.name:
        raise ValueError("run name must be a stem")
    validate_settings(json.loads((EXPERIMENT / "config.json").read_text()))
    asyncio.run(run(args.name, args.retired_parent, args.source_bootstrap_dir))


if __name__ == "__main__":
    main()
