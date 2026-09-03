#!/usr/bin/env python3
"""Render the frozen held-out ID trajectories with the student prompt."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from experiments.tool_trajectory_monitoring.prepare_distillation_ood import (
    atomic_write_json,
    atomic_write_jsonl,
    materialize_rows,
    read_jsonl,
    sha256_file,
)
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set

EXPECTED_ROWS = 3_012
EXPECTED_COUNTS = {
    "gloom_exfiltration:0": 1_035,
    "gloom_exfiltration:1": 1_031,
    "test_stride:0": 369,
    "test_stride:1": 577,
}
DEFAULT_SOURCE = Path(
    "data/tool_trajectory_monitoring/teacher_id_benchmark/prompts.jsonl"
)
DEFAULT_OUTPUT = Path(
    "data/tool_trajectory_monitoring/distillation_id/prompts.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = read_jsonl(args.source)
    rows = materialize_rows(source_rows, expected_rows=EXPECTED_ROWS)
    counts = Counter(
        (row["metadata"]["source_dataset"], row["metadata"]["ground_truth"])
        for row in rows
    )
    rendered_counts = {
        f"{source}:{label}": count
        for (source, label), count in sorted(counts.items())
    }
    if rendered_counts != EXPECTED_COUNTS:
        raise ValueError(f"held-out ID counts drifted: {rendered_counts}")

    atomic_write_jsonl(args.output, rows)
    prompts = load_prompt_set()
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    atomic_write_json(
        manifest_path,
        {
            "campaign_id": "gleipnir4b-held-out-id-v1",
            "source": args.source.as_posix(),
            "source_sha256": sha256_file(args.source),
            "output": args.output.as_posix(),
            "output_sha256": sha256_file(args.output),
            "rows": len(rows),
            "source_label_counts": rendered_counts,
            "prompt_set_id": prompts.prompt_set_id,
            "teacher_template_sha256": prompts.teacher.template_sha256,
            "student_template_sha256": prompts.student.template_sha256,
            "transformation": (
                "Validate and extract each frozen held-out ID trajectory from "
                "the teacher prompt, then rerender it under the compact student "
                "prompt without changing membership, labels, or trajectory text."
            ),
        },
    )
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
