#!/usr/bin/env python3
"""Materialize and token-count the paper's exact teacher-training mixture."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from experiments.tool_trajectory_monitoring.teacher_canary import (
    atomic_write_json,
    atomic_write_jsonl,
    sha256_file,
)

SOURCE_REPO = "adityaasinha28/deliberative-monitor-pipeline"
SOURCE_REVISION = "2f10f4e49f4bf349965640ac021406689455d1c3"
SOURCE_FILE = "merged/stride_gloom_crh_s45_o46_ctrl_t7_bal.parquet"
SOURCE_SHA256 = "43bfc9f78d6584c43135109636f5e1747c694842419a327c3f37614fb9301802"
SOURCE_LICENSE = "unspecified_by_upstream"

FILTER_TOKENIZER_ID = "Qwen/Qwen3-8B"
FILTER_TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
KIMI_TOKENIZER_ID = "moonshotai/Kimi-K3"
KIMI_TOKENIZER_REVISION = "a590ce090cb049c93a33dfe8c208ec652aa20503"
DEFAULT_MAX_INPUT_TOKENS = 28_000
MAKORA_CHAT_WRAPPER_TOKENS = 21

RAW_SOURCE_COUNTS = {
    "stride": {0: 675, 1: 675},
    "gloom": {0: 386, 1: 386},
    "crh": {0: 1_170, 1: 1_170},
    "s45": {0: 885, 1: 885},
    "o46": {0: 510, 1: 510},
    "ctrl": {0: 718, 1: 718},
}
PAPER_SOURCE_MAP = {
    "stride": "stride",
    "gloom": "gloom",
    "crh": "cot_red_handed",
    "s45": "bash_arena",
    "o46": "bash_arena",
    "ctrl": "bash_bench",
}
PAPER_SOURCE_COUNTS = {
    "stride": {0: 675, 1: 675},
    "gloom": {0: 386, 1: 386},
    "cot_red_handed": {0: 1_170, 1: 1_170},
    "bash_arena": {0: 1_395, 1: 1_395},
    "bash_bench": {0: 718, 1: 718},
}

DEFAULT_SOURCE = Path(
    "data/tool_trajectory_monitoring/source/deliberative-monitor-pipeline-2f10f4e/"
    "merged/stride_gloom_crh_s45_o46_ctrl_t7_bal.parquet"
)
DEFAULT_OUTPUT = Path(
    "data/tool_trajectory_monitoring/teacher_training/prompts.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS
    )
    return parser.parse_args()


def extract_trajectory(messages: Any) -> str:
    """Extract the action-only user payload from the paper's SFT message row."""
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if roles != ["system", "user", "assistant"] or len(messages) != 3:
        raise ValueError(f"unexpected paper message roles: {roles}")
    trajectory = messages[1].get("content")
    if not isinstance(trajectory, str) or not trajectory.strip():
        raise ValueError("paper trajectory is empty")
    return trajectory


def expected_counter(counts: dict[str, dict[int, int]]) -> Counter[tuple[str, int]]:
    return Counter(
        {
            (source, label): count
            for source, label_counts in counts.items()
            for label, count in label_counts.items()
        }
    )


def validate_counts(
    rows: list[dict[str, Any]],
    *,
    source_key: str,
    expected: dict[str, dict[int, int]],
) -> None:
    actual = Counter((str(row[source_key]), int(row["ground_truth"])) for row in rows)
    target = expected_counter(expected)
    if actual != target:
        raise ValueError(f"{source_key} counts drifted: {actual} != {target}")


def add_token_lengths(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    text_key: str,
    output_key: str,
    batch_size: int = 16,
) -> None:
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [str(row[text_key]) for row in batch],
            add_special_tokens=False,
            return_length=True,
        )
        for row, length in zip(batch, encoded["length"], strict=True):
            row[output_key] = int(length)


def length_summary(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        raise ValueError("cannot summarize empty token lengths")
    values = np.asarray(lengths, dtype=np.int64)
    return {
        "total": int(values.sum()),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": int(values.max()),
    }


def source_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for source in PAPER_SOURCE_COUNTS:
        selected = [row for row in rows if row["source_dataset"] == source]
        output[source] = {
            "rows": len(selected),
            "labels": dict(
                sorted(Counter(str(row["ground_truth"]) for row in selected).items())
            ),
            "paper_input_tokens": length_summary(
                [int(row["paper_input_tokens"]) for row in selected]
            ),
            "kimi_raw_prompt_tokens": length_summary(
                [int(row["kimi_raw_prompt_tokens"]) for row in selected]
            ),
            "predicted_provider_prompt_tokens": length_summary(
                [int(row["predicted_provider_prompt_tokens"]) for row in selected]
            ),
        }
    return output


def main() -> None:
    args = parse_args()
    if args.max_input_tokens < 1:
        raise ValueError("max input tokens must be positive")
    actual_sha256 = sha256_file(args.source)
    if actual_sha256 != SOURCE_SHA256:
        raise ValueError(
            f"source checksum drift for {args.source}: {actual_sha256}"
        )

    table = pq.read_table(
        args.source,
        columns=[
            "messages",
            "prompt_id",
            "ground_truth",
            "dataset_source",
            "sample_index",
            "original_prompt_id",
        ],
    )
    rows = []
    for source_row_index, source_row in enumerate(table.to_pylist()):
        raw_source = str(source_row["dataset_source"])
        if raw_source not in PAPER_SOURCE_MAP:
            raise ValueError(f"unknown paper source: {raw_source!r}")
        rows.append(
            {
                "source_row_index": source_row_index,
                "raw_source": raw_source,
                "source_dataset": PAPER_SOURCE_MAP[raw_source],
                "ground_truth": int(source_row["ground_truth"]),
                "prompt_id": int(source_row["prompt_id"]),
                "original_prompt_id": int(source_row["original_prompt_id"]),
                "sample_index": int(source_row["sample_index"]),
                "trajectory": extract_trajectory(source_row["messages"]),
            }
        )
    validate_counts(rows, source_key="raw_source", expected=RAW_SOURCE_COUNTS)
    validate_counts(rows, source_key="source_dataset", expected=PAPER_SOURCE_COUNTS)

    filter_tokenizer = AutoTokenizer.from_pretrained(
        FILTER_TOKENIZER_ID,
        revision=FILTER_TOKENIZER_REVISION,
    )
    add_token_lengths(
        rows,
        filter_tokenizer,
        text_key="trajectory",
        output_key="paper_input_tokens",
    )
    eligible = [
        row
        for row in rows
        if int(row["paper_input_tokens"]) <= args.max_input_tokens
    ]
    validate_counts(
        eligible,
        source_key="source_dataset",
        expected=PAPER_SOURCE_COUNTS,
    )

    prompt_set = load_prompt_set()
    template = prompt_set.teacher
    cache_prefix_sha256 = hashlib.sha256(
        template.cache_prefix.encode("utf-8")
    ).hexdigest()
    for row in eligible:
        row["prompt"] = template.render(str(row["trajectory"]))

    kimi_tokenizer = AutoTokenizer.from_pretrained(
        KIMI_TOKENIZER_ID,
        revision=KIMI_TOKENIZER_REVISION,
        trust_remote_code=True,
    )
    add_token_lengths(
        eligible,
        kimi_tokenizer,
        text_key="prompt",
        output_key="kimi_raw_prompt_tokens",
    )

    output_rows = []
    for row in eligible:
        prompt = str(row["prompt"])
        raw_source = str(row["raw_source"])
        source_identity = (
            f"{SOURCE_REVISION}\0{raw_source}\0{row['source_row_index']}\0"
            f"{row['original_prompt_id']}"
        )
        provider_tokens = (
            int(row["kimi_raw_prompt_tokens"]) + MAKORA_CHAT_WRAPPER_TOKENS
        )
        row["predicted_provider_prompt_tokens"] = provider_tokens
        output_rows.append(
            {
                "id": hashlib.sha256(source_identity.encode()).hexdigest()[:24],
                "prompt": prompt,
                "metadata": {
                    "source_repo": SOURCE_REPO,
                    "source_revision": SOURCE_REVISION,
                    "source_file": SOURCE_FILE,
                    "source_license": SOURCE_LICENSE,
                    "source_row_index": row["source_row_index"],
                    "source_dataset": row["source_dataset"],
                    "raw_source": raw_source,
                    "ground_truth": row["ground_truth"],
                    "ground_truth_provenance": "paper_author_final_sft_artifact",
                    "prompt_id": row["prompt_id"],
                    "original_prompt_id": row["original_prompt_id"],
                    "sample_index": row["sample_index"],
                    "trajectory_sha256": hashlib.sha256(
                        str(row["trajectory"]).encode("utf-8")
                    ).hexdigest(),
                    "paper_input_tokens": row["paper_input_tokens"],
                    "paper_filter_tokenizer_id": FILTER_TOKENIZER_ID,
                    "paper_filter_tokenizer_revision": FILTER_TOKENIZER_REVISION,
                    "kimi_raw_prompt_tokens": row["kimi_raw_prompt_tokens"],
                    "predicted_provider_prompt_tokens": provider_tokens,
                    "provider_chat_wrapper_tokens": MAKORA_CHAT_WRAPPER_TOKENS,
                    "prompt_set_id": prompt_set.prompt_set_id,
                    "prompt_template_sha256": template.template_sha256,
                    "cache_prefix_chars": len(template.cache_prefix),
                    "cache_prefix_sha256": cache_prefix_sha256,
                    "rendered_prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                },
            }
        )

    atomic_write_jsonl(args.output, output_rows)
    all_summary = {
        "rows": len(eligible),
        "paper_input_tokens": length_summary(
            [int(row["paper_input_tokens"]) for row in eligible]
        ),
        "kimi_raw_prompt_tokens": length_summary(
            [int(row["kimi_raw_prompt_tokens"]) for row in eligible]
        ),
        "predicted_provider_prompt_tokens": length_summary(
            [int(row["predicted_provider_prompt_tokens"]) for row in eligible]
        ),
    }
    manifest = {
        "source": {
            "repo": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "file": SOURCE_FILE,
            "sha256": SOURCE_SHA256,
            "license": SOURCE_LICENSE,
            "usage": "internal_research_only; do_not_redistribute",
        },
        "selection": {
            "rule": "paper_author_final_8688_mix_and_input_tokens_lte_28000",
            "max_input_tokens": args.max_input_tokens,
            "filter_tokenizer_id": FILTER_TOKENIZER_ID,
            "filter_tokenizer_revision": FILTER_TOKENIZER_REVISION,
            "candidate_rows": len(rows),
            "eligible_rows": len(eligible),
            "filtered_rows": len(rows) - len(eligible),
            "raw_source_counts": RAW_SOURCE_COUNTS,
            "paper_source_counts": PAPER_SOURCE_COUNTS,
        },
        "prompt": {
            "prompt_set_id": prompt_set.prompt_set_id,
            "template_sha256": template.template_sha256,
            "cache_prefix_sha256": cache_prefix_sha256,
            "cache_prefix_chars": len(template.cache_prefix),
        },
        "token_accounting": {
            "kimi_tokenizer_id": KIMI_TOKENIZER_ID,
            "kimi_tokenizer_revision": KIMI_TOKENIZER_REVISION,
            "provider": "Makora",
            "provider_chat_wrapper_tokens_per_row": MAKORA_CHAT_WRAPPER_TOKENS,
            "provider_wrapper_basis": "exact difference on paid canary rows",
            "all": all_summary,
            "by_source": source_summaries(eligible),
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
            "rows": len(output_rows),
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    atomic_write_json(manifest_path, manifest)
    print(
        f"materialized rows={len(output_rows)} "
        f"provider_prompt_tokens="
        f"{all_summary['predicted_provider_prompt_tokens']['total']} "
        f"output={args.output} manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
