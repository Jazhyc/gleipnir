"""Compute AUROC and request diagnostics for a Kimi teacher canary cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.tool_trajectory_monitoring.teacher_canary import (
    DEFAULT_SEED,
    atomic_write_json,
    load_jsonl,
    summarize_scored_rows,
)

DEFAULT_INPUT = Path("data/tool_trajectory_monitoring/canary/prompts.jsonl")
DEFAULT_SCORES = Path("results/tool_trajectory_monitoring/kimi_k3_canary.jsonl")
DEFAULT_OUTPUT = Path("results/tool_trajectory_monitoring/kimi_k3_canary_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--scores",
        type=Path,
        action="append",
        help="score JSONL; repeat for separately configured recovery caches",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-resamples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_paths = args.scores or [DEFAULT_SCORES]
    score_rows = [row for path in score_paths for row in load_jsonl(path)]
    summary = summarize_scored_rows(
        load_jsonl(args.input),
        score_rows,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    atomic_write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"score inputs: {', '.join(str(path) for path in score_paths)}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
