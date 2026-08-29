#!/usr/bin/env python3
"""Materialize the frozen Qwen reasoning canary from the teacher selection."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from experiments.tool_trajectory_monitoring.qwen_reasoning_core import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    load_jsonl,
)
from experiments.tool_trajectory_monitoring.teacher_canary import sha256_file

DEFAULT_SELECTION = Path("data/tool_trajectory_monitoring/canary/prompts.jsonl")
DEFAULT_SOURCE_DIRECTORY = Path(
    "data/tool_trajectory_monitoring/source/blackboxmonitorsMATS-ccb77e3"
)
DEFAULT_OUTPUT = Path("data/tool_trajectory_monitoring/qwen_reasoning/records.jsonl")


def source_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(
        path,
        columns=["id", "ground_truth", "formatted_transcript", "source_dataset"],
    ).to_pylist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument(
        "--source-directory", type=Path, default=DEFAULT_SOURCE_DIRECTORY
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "experiments/tool_trajectory_monitoring/qwen_reasoning_benchmark.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    selections = load_jsonl(args.selection)
    expected_rows = int(config["scope"]["rows"])
    if len(selections) != expected_rows:
        raise ValueError(
            f"selection has {len(selections)} rows; config requires {expected_rows}"
        )
    files = {
        str(row["metadata"]["source_file"])
        for row in selections
    }
    loaded = {
        filename: source_rows(args.source_directory / filename)
        for filename in sorted(files)
    }
    output_rows = []
    for selection in selections:
        metadata = selection["metadata"]
        filename = str(metadata["source_file"])
        row_index = int(metadata["source_row_index"])
        source = loaded[filename][row_index]
        trajectory = str(source["formatted_transcript"])
        checks = {
            "source_id": str(source["id"]) == str(metadata["source_id"]),
            "source": str(source["source_dataset"])
            == str(metadata["source_dataset"]),
            "label": int(source["ground_truth"])
            == int(metadata["ground_truth"]),
            "trajectory": hashlib.sha256(trajectory.encode()).hexdigest()
            == str(metadata["trajectory_sha256"]),
        }
        if not all(checks.values()):
            failed = sorted(key for key, passed in checks.items() if not passed)
            raise ValueError(f"source drift for id={selection['id']}: {failed}")
        output_rows.append(
            {
                "id": str(selection["id"]),
                "source": str(source["source_dataset"]),
                "label": int(source["ground_truth"]),
                "trajectory": trajectory,
                "trajectory_sha256": str(metadata["trajectory_sha256"]),
                "source_file": filename,
                "source_row_index": row_index,
                "source_id": str(source["id"]),
                "source_input_tokens": int(metadata["input_tokens"]),
            }
        )
    if len({row["id"] for row in output_rows}) != len(output_rows):
        raise ValueError("selection contains duplicate IDs")
    atomic_write_jsonl(args.output, output_rows)
    manifest = {
        "campaign_id": config["campaign_id"],
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "selection_path": str(args.selection),
        "selection_sha256": sha256_file(args.selection),
        "source_revision": config["scope"]["source_revision"],
        "source_files": {
            filename: {
                "sha256": sha256_file(args.source_directory / filename),
                "bytes": (args.source_directory / filename).stat().st_size,
            }
            for filename in sorted(files)
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
            "rows": len(output_rows),
        },
    }
    atomic_write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(f"materialized rows={len(output_rows)} output={args.output}")


if __name__ == "__main__":
    main()
