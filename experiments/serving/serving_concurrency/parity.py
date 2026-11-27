"""Score the frozen ID and OOD workloads on one 4B serving step.

At concurrency 128 this is ``experiments.release_serving.validate`` in the
recorded campaign's batching. Steps there hold many requests, so graphs of up
to a few thousand tokens are rarely replayed; ``--concurrency 1`` sends one
request at a time, where they are.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

from experiments.serving.serving_concurrency.run import OUT, PORT, ROOT, launch, prepare


def score_serial(out: Path, concurrency: int, splits: list[str] | None) -> None:
    """Score both splits at ``concurrency`` and report source-macro pAUROC@20."""
    from experiments.release_serving import validate
    from experiments.release_serving.targets import TARGETS
    from gleipnir.data.monitoring import read_rows, write_json, write_rows
    from gleipnir.evaluation.http_score import score_batch

    config = {"port": PORT, "concurrency": concurrency, "timeout_seconds": 600}
    report = {"concurrency": concurrency, "splits": {}}
    for split, (workload_path, recorded_path, _) in TARGETS["4b"].validation.items():
        if splits and split not in splits:
            continue
        workload = validate.rows(workload_path)
        recorded = {r["id"]: float(r["score"]) for r in read_rows(ROOT / recorded_path)}
        scored, seconds = asyncio.run(score_batch(workload, config))
        write_rows(
            out / f"{split}.jsonl",
            [
                {
                    "id": r["id"],
                    "score": r["score"],
                    "margin": r["margin"],
                    "prompt_tokens": r["prompt_tokens"],
                }
                for r in scored
            ],
        )
        new = {r["id"]: float(r["score"]) for r in scored}
        truth = validate.labels(split)
        tokens = sum(r["prompt_tokens"] for r in scored)
        report["splits"][split] = {
            "rows": len(new),
            "pauroc_at_20": validate.macro_pauroc(new, truth, split),
            "recorded_pauroc_at_20": validate.macro_pauroc(recorded, truth, split),
            "seconds": seconds,
            "prompt_tokens_per_second": tokens / seconds,
        }
        print({split: report["splits"][split]}, flush=True)
    write_json(out / "report.json", report)


def run(
    step: str, session: str, concurrency: int, splits: list[str] | None = None
) -> None:
    from gleipnir.data.monitoring import write_json
    from gleipnir.serving import bf16_launch

    name = f"parity_4b_{step}" + ("" if concurrency == 128 else f"_c{concurrency}")
    out = OUT / session / name
    out.mkdir(parents=True, exist_ok=False)
    command, env = prepare("4b", step, out)
    log = f"{session}_{name}".lstrip("_").replace("/", "_")
    process, startup, attempt = launch(command, env, log, out)
    try:
        code = 0
        if concurrency == 128:
            validate = [sys.executable, "-m", "experiments.release_serving.validate"]
            validate += ["4b", "--port", str(PORT), "--rows-output", str(out)]
            validate += ["--output", str(out / "report.json")]
            code = subprocess.run(validate, cwd=ROOT, check=False).returncode
        else:
            score_serial(out, concurrency, splits)
        write_json(
            out / "condition.json",
            {
                "step": step,
                "concurrency": concurrency,
                "command": command,
                "startup_seconds": startup,
                "startup_attempt": attempt,
                "validate_exit": code,
            },
        )
    finally:
        bf16_launch.stop(process)
    print(f"parity_complete {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True)
    parser.add_argument("--session", default="", help="output subdirectory")
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--splits", nargs="+", choices=("id", "ood"))
    args = parser.parse_args()
    run(args.step, args.session, args.concurrency, args.splits)


if __name__ == "__main__":
    main()
