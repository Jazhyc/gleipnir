#!/usr/bin/env python3
"""Render and token-count Kimi prompts for the held-out ID benchmark."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from experiments.tool_trajectory_monitoring.prepare_teacher_training_cache import (
    DEFAULT_MAX_INPUT_TOKENS,
    FILTER_TOKENIZER_ID,
    FILTER_TOKENIZER_REVISION,
    KIMI_TOKENIZER_ID,
    KIMI_TOKENIZER_REVISION,
    MAKORA_CHAT_WRAPPER_TOKENS,
    add_token_lengths,
    length_summary,
)
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    load_jsonl,
)
from gleipnir.qwen35_adapter_rebase import sha256_file

INPUT_SHA256 = "6625ddba6a266866115db64c844c32a07ab4fcb6cd236322ebc6e17343c58cb4"
INPUT_MANIFEST_SHA256 = (
    "40cb437f4d1dc31dc12a5f916925f52c00dd54c8b3abfc203caa9c2b1788b921"
)
EXPECTED_COUNTS = {
    "test_stride": {0: 369, 1: 577},
    "gloom_exfiltration": {0: 1_035, 1: 1_031},
}
DEFAULT_INPUT = Path(
    "data/tool_trajectory_monitoring/qwen_reasoning_id/records.jsonl"
)
DEFAULT_OUTPUT = Path(
    "data/tool_trajectory_monitoring/teacher_id_benchmark/prompts.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS
    )
    return parser.parse_args()


def validate_id_rows(rows: list[dict[str, Any]]) -> None:
    actual = Counter((str(row.get("source")), int(row.get("label"))) for row in rows)
    expected = Counter(
        {
            (source, label): count
            for source, label_counts in EXPECTED_COUNTS.items()
            for label, count in label_counts.items()
        }
    )
    if actual != expected:
        raise ValueError(
            f"held-out source-label counts drifted: {actual} != {expected}"
        )
    ids = [str(row.get("id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("held-out rows contain duplicate IDs")
    for row in rows:
        trajectory = row.get("trajectory")
        if not isinstance(trajectory, str) or not trajectory.strip():
            raise ValueError(f"held-out trajectory is empty for id={row.get('id')!r}")


def source_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for source in EXPECTED_COUNTS:
        selected = [row for row in rows if row["source"] == source]
        output[source] = {
            "rows": len(selected),
            "labels": dict(
                sorted(Counter(str(row["label"]) for row in selected).items())
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
    manifest_path = args.input.with_suffix(".manifest.json")
    if sha256_file(args.input) != INPUT_SHA256:
        raise ValueError(f"held-out input checksum drifted: {args.input}")
    if sha256_file(manifest_path) != INPUT_MANIFEST_SHA256:
        raise ValueError(f"held-out input manifest checksum drifted: {manifest_path}")
    input_manifest = load_json(manifest_path)
    rows = load_jsonl(args.input)
    validate_id_rows(rows)

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
    over_limit = [
        str(row["id"])
        for row in rows
        if int(row["paper_input_tokens"]) > args.max_input_tokens
    ]
    if over_limit:
        raise ValueError(
            f"{len(over_limit)} held-out rows exceed the paper input cap; "
            f"first={over_limit[0]!r}"
        )

    prompt_set = load_prompt_set()
    template = prompt_set.teacher
    cache_prefix_sha256 = hashlib.sha256(
        template.cache_prefix.encode("utf-8")
    ).hexdigest()
    for row in rows:
        row["prompt"] = template.render(str(row["trajectory"]))

    kimi_tokenizer = AutoTokenizer.from_pretrained(
        KIMI_TOKENIZER_ID,
        revision=KIMI_TOKENIZER_REVISION,
        trust_remote_code=True,
    )
    add_token_lengths(
        rows,
        kimi_tokenizer,
        text_key="prompt",
        output_key="kimi_raw_prompt_tokens",
    )

    output_rows = []
    for row in rows:
        prompt = str(row["prompt"])
        provider_tokens = (
            int(row["kimi_raw_prompt_tokens"]) + MAKORA_CHAT_WRAPPER_TOKENS
        )
        row["predicted_provider_prompt_tokens"] = provider_tokens
        output_rows.append(
            {
                "id": row["id"],
                "prompt": prompt,
                "metadata": {
                    "source_dataset": row["source"],
                    "ground_truth": row["label"],
                    "ground_truth_provenance": "held_out_source_dataset",
                    "source_repo": row["source_repo"],
                    "source_revision": row["source_revision"],
                    "source_file": row["source_file"],
                    "source_row_index": row["source_row_index"],
                    "source_id": row["source_id"],
                    "trajectory_sha256": row["trajectory_sha256"],
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
    token_accounting = {
        "kimi_tokenizer_id": KIMI_TOKENIZER_ID,
        "kimi_tokenizer_revision": KIMI_TOKENIZER_REVISION,
        "provider": "Makora",
        "provider_chat_wrapper_tokens_per_row": MAKORA_CHAT_WRAPPER_TOKENS,
        "provider_wrapper_basis": "exact difference on paid canary rows",
        "all": {
            "rows": len(rows),
            "paper_input_tokens": length_summary(
                [int(row["paper_input_tokens"]) for row in rows]
            ),
            "kimi_raw_prompt_tokens": length_summary(
                [int(row["kimi_raw_prompt_tokens"]) for row in rows]
            ),
            "predicted_provider_prompt_tokens": length_summary(
                [int(row["predicted_provider_prompt_tokens"]) for row in rows]
            ),
        },
        "by_source": source_summaries(rows),
    }
    output_manifest = {
        "input": {
            "path": str(args.input),
            "sha256": INPUT_SHA256,
            "manifest_path": str(manifest_path),
            "manifest_sha256": INPUT_MANIFEST_SHA256,
            "sources": input_manifest["sources"],
            "usage": input_manifest["usage"],
        },
        "selection": {
            "role": "held_out_in_distribution_validation",
            "max_input_tokens": args.max_input_tokens,
            "filter_tokenizer_id": FILTER_TOKENIZER_ID,
            "filter_tokenizer_revision": FILTER_TOKENIZER_REVISION,
            "rows": len(rows),
            "filtered_rows": 0,
            "source_label_counts": EXPECTED_COUNTS,
        },
        "prompt": {
            "prompt_set_id": prompt_set.prompt_set_id,
            "template_sha256": template.template_sha256,
            "cache_prefix_sha256": cache_prefix_sha256,
            "cache_prefix_chars": len(template.cache_prefix),
        },
        "token_accounting": token_accounting,
        "output": {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
            "rows": len(output_rows),
        },
    }
    output_manifest_path = args.output.with_suffix(".manifest.json")
    atomic_write_json(output_manifest_path, output_manifest)
    total = token_accounting["all"]["predicted_provider_prompt_tokens"]["total"]
    print(
        f"materialized rows={len(output_rows)} provider_prompt_tokens={total} "
        f"output={args.output} manifest={output_manifest_path}"
    )


if __name__ == "__main__":
    main()
