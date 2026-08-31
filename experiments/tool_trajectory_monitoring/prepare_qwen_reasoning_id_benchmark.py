#!/usr/bin/env python3
"""Materialize the paper's held-out STRIDE and Gloom-Exfiltration sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from experiments.tool_trajectory_monitoring.qwen_reasoning_core import (
    atomic_write_json,
    atomic_write_jsonl,
)
from gleipnir.qwen35_adapter_rebase import sha256_file

STRIDE_REPO = "aksh-n/rich-trajectories-cleaned"
STRIDE_REVISION = "92a44c828935c325dc86cb390f874d0ee9345afb"
STRIDE_FILE = "data/test-00000-of-00001.parquet"
STRIDE_SHA256 = "c13af47d00d3a32e9fcecb13df01e9f8bdf6a80e956d25ed00e8cb15dcc1d43b"
GLOOM_REPO = "Syghmon/gloom-data-exfiltration"
GLOOM_REVISION = "ee3c56ba4c12e213f4f354e54021b13c6b448365"
GLOOM_FILE = "data/train-00000-of-00001.parquet"
GLOOM_SHA256 = "5c0c7a57c6d6ffd6d55151219a7d9c4050ffd40c956bf8c9f2c6cf07d34b898c"
GLOOM_PREDICTION_REPO = "aksh-n/deliberative-monitor-benchmark"
GLOOM_PREDICTION_REVISION = "b45f3a8bb08e3e1e520fb49ce1b81a63fad8c80d"
GLOOM_MAX_RAW_TOKENS = 27_900
GLOOM_SELECTED_IDS_SHA256 = (
    "f333ba46cab4520ba4357a3935a92c1273565d7ad1ec49902c1ec0926134b7a5"
)
TOKENIZER_ID = "Qwen/Qwen3-8B"
TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EXPECTED_COUNTS = {
    "test_stride": {0: 369, 1: 577},
    "gloom_exfiltration": {0: 1_035, 1: 1_031},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-directory",
        type=Path,
        default=Path("data/tool_trajectory_monitoring/source/id_evaluation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tool_trajectory_monitoring/qwen_reasoning_id/records.jsonl"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "experiments/tool_trajectory_monitoring/qwen_reasoning_id_benchmark.json"
        ),
    )
    return parser.parse_args()


def single_user_content(value: str) -> str:
    """Extract one serialized user message without accepting schema drift."""
    messages = json.loads(value)
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("expected exactly one serialized message")
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        raise ValueError("expected one serialized user message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("serialized user content is empty")
    return content


def add_token_counts(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    batch_size: int = 16,
) -> None:
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [str(row["trajectory"]) for row in batch],
            add_special_tokens=False,
            return_length=True,
        )
        for row, length in zip(batch, encoded["length"], strict=True):
            row["source_tokens"] = int(length)


def selected_id_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(str(row["source_id"]) for row in rows) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_counts(rows: list[dict[str, Any]], source: str) -> None:
    actual = Counter(int(row["label"]) for row in rows)
    expected = EXPECTED_COUNTS[source]
    if actual != Counter(expected):
        raise ValueError(f"{source} label counts drifted: {dict(actual)} != {expected}")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError(f"{source} contains duplicate IDs")


def load_stride(path: Path) -> list[dict[str, Any]]:
    rows = pq.read_table(
        path,
        columns=["id", "prompt", "ground_truth", "extra_info"],
    ).to_pylist()
    output = []
    for source_row_index, row in enumerate(rows):
        extra_info = row["extra_info"]
        if not isinstance(extra_info, dict) or extra_info.get("split") != "test":
            raise ValueError("STRIDE test split metadata drifted")
        messages = row["prompt"]
        if not isinstance(messages, list) or len(messages) != 1:
            raise ValueError("STRIDE prompt must contain one user message")
        message = messages[0]
        if not isinstance(message, dict) or message.get("role") != "user":
            raise ValueError("STRIDE prompt must contain one user message")
        trajectory = message.get("content")
        if not isinstance(trajectory, str) or not trajectory.strip():
            raise ValueError("STRIDE trajectory is empty")
        source_id = str(row["id"])
        output.append(
            {
                "id": f"test_stride:{source_id}",
                "source": "test_stride",
                "label": int(row["ground_truth"]),
                "trajectory": trajectory,
                "trajectory_sha256": hashlib.sha256(trajectory.encode()).hexdigest(),
                "source_repo": STRIDE_REPO,
                "source_revision": STRIDE_REVISION,
                "source_file": STRIDE_FILE,
                "source_row_index": source_row_index,
                "source_id": source_id,
                "source_tokens": None,
            }
        )
    validate_counts(output, "test_stride")
    return output


def load_gloom(path: Path, tokenizer: Any) -> list[dict[str, Any]]:
    rows = pq.read_table(
        path,
        columns=["id", "prompt", "ground_truth", "transcript"],
    ).to_pylist()
    candidates = []
    for source_row_index, row in enumerate(rows):
        trajectory = str(row["transcript"])
        if trajectory != single_user_content(str(row["prompt"])):
            raise ValueError("Gloom prompt and action-only transcript differ")
        source_id = str(row["id"])
        candidates.append(
            {
                "id": f"gloom_exfiltration:{source_id}",
                "source": "gloom_exfiltration",
                "label": int(row["ground_truth"]),
                "trajectory": trajectory,
                "trajectory_sha256": hashlib.sha256(trajectory.encode()).hexdigest(),
                "source_repo": GLOOM_REPO,
                "source_revision": GLOOM_REVISION,
                "source_file": GLOOM_FILE,
                "source_row_index": source_row_index,
                "source_id": source_id,
            }
        )
    add_token_counts(candidates, tokenizer)
    selected = [
        row for row in candidates if int(row["source_tokens"]) <= GLOOM_MAX_RAW_TOKENS
    ]
    validate_counts(selected, "gloom_exfiltration")
    digest = selected_id_sha256(selected)
    if digest != GLOOM_SELECTED_IDS_SHA256:
        raise ValueError(f"Gloom selected-ID digest drifted: {digest}")
    return selected


def main() -> None:
    args = parse_args()
    stride_path = args.source_directory / "stride_test.parquet"
    gloom_path = args.source_directory / "gloom_exfiltration.parquet"
    expected_hashes = {
        stride_path: STRIDE_SHA256,
        gloom_path: GLOOM_SHA256,
    }
    for path, expected in expected_hashes.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"source checksum drift for {path}: {actual}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        revision=TOKENIZER_REVISION,
    )
    stride = load_stride(stride_path)
    gloom = load_gloom(gloom_path, tokenizer)
    output_rows = stride + gloom
    atomic_write_jsonl(args.output, output_rows)
    manifest = {
        "campaign_id": "qwen35-9b-reasoning-id-v1",
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "sources": {
            "test_stride": {
                "repo": STRIDE_REPO,
                "revision": STRIDE_REVISION,
                "file": STRIDE_FILE,
                "sha256": STRIDE_SHA256,
                "rows": len(stride),
                "license": "unspecified_by_upstream",
            },
            "gloom_exfiltration": {
                "repo": GLOOM_REPO,
                "revision": GLOOM_REVISION,
                "file": GLOOM_FILE,
                "sha256": GLOOM_SHA256,
                "raw_rows": 2_190,
                "selected_rows": len(gloom),
                "selection": {
                    "tokenizer": TOKENIZER_ID,
                    "tokenizer_revision": TOKENIZER_REVISION,
                    "max_raw_tokens": GLOOM_MAX_RAW_TOKENS,
                    "selected_ids_sha256": GLOOM_SELECTED_IDS_SHA256,
                    "label_sequence_verified_against": {
                        "repo": GLOOM_PREDICTION_REPO,
                        "revision": GLOOM_PREDICTION_REVISION,
                    },
                },
                "license": "unspecified_by_upstream",
            },
        },
        "usage": "internal_research_evaluation_only; do_not_redistribute",
        "output": {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
            "rows": len(output_rows),
            "counts": {
                source: {str(label): count for label, count in counts.items()}
                for source, counts in EXPECTED_COUNTS.items()
            },
        },
    }
    atomic_write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(
        f"materialized rows={len(output_rows)} stride={len(stride)} "
        f"gloom_exfiltration={len(gloom)} output={args.output}"
    )


if __name__ == "__main__":
    main()
