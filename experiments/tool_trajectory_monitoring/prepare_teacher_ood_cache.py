#!/usr/bin/env python3
"""Render and token-count Kimi prompts for the paper author's OOD suite."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from experiments.tool_trajectory_monitoring.prepare_teacher_training_cache import (
    KIMI_TOKENIZER_ID,
    KIMI_TOKENIZER_REVISION,
    add_token_lengths,
    length_summary,
)
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from experiments.tool_trajectory_monitoring.teacher_canary import (
    atomic_write_json,
    atomic_write_jsonl,
    sha256_file,
)

SOURCE_REPO = "adityaasinha28/deliberative-monitor-pipeline"
SOURCE_REVISION = "2f10f4e49f4bf349965640ac021406689455d1c3"
SOURCE_LICENSE = "unspecified_by_upstream"
SOURCE_DIRECTORY = Path(
    "data/tool_trajectory_monitoring/source/deliberative-monitor-pipeline-2f10f4e/"
    "filtered_eval"
)
DEFAULT_OUTPUT = Path(
    "data/tool_trajectory_monitoring/teacher_ood_benchmark/prompts.jsonl"
)
FILTER_LOG_FILE = "filtered_eval/filter_log.json"
FILTER_LOG_SHA256 = "97633ac074b331e7789e645fb03a9aeb2e0760b76f9f318de75119d03da0f5f4"

# OpenRouter usage differed from raw Kimi tokenization by exactly 21 prompt
# tokens on all 3,312 prior Fireworks requests.
FIREWORKS_CHAT_WRAPPER_TOKENS = 21
FIREWORKS_INPUT_USD_PER_MILLION = 3.00
FIREWORKS_OUTPUT_USD_PER_MILLION = 15.00
FIREWORKS_COMPLETION_TOKENS_PER_ROW = 10


@dataclass(frozen=True)
class OODSource:
    source: str
    filename: str
    sha256: str
    labels: dict[int, int]

    @property
    def repo_file(self) -> str:
        return f"filtered_eval/{self.filename}"


OOD_SOURCES = (
    OODSource(
        "cua_shade_arena",
        "cua_shade.parquet",
        "53b1adf597dc226249f1b0d2111cd079898d45608c955747ea2613d27092b19e",
        {0: 479, 1: 476},
    ),
    OODSource(
        "mle_sabotage",
        "mle_sabotage.parquet",
        "133388e49e3837c059e76e6140349889cfde6ccd039bcd5cb89e507ea592c2b0",
        {0: 50, 1: 98},
    ),
    OODSource(
        "agentdojo",
        "ctrl_agentdojo.parquet",
        "9d0c7024826eb78357c45fea852225219788533bb7617a79017afd2530a14608",
        {0: 750, 1: 750},
    ),
    OODSource(
        "iac",
        "ctrl_iac.parquet",
        "f57d279b10e9b3eabf8c35dd0340d7799d861b5bb851280391413c6249105d42",
        {0: 728, 1: 715},
    ),
    OODSource(
        "rogue_eval",
        "ctrl_rogue_eval.parquet",
        "937acc7e2f2a99958b1e450a06d210fcad292f10f09d2e8fa74b8c414f50c3db",
        {0: 750, 1: 749},
    ),
    OODSource(
        "shade_arena",
        "ctrl_shade.parquet",
        "1a7aac49a5cca05a2a30ef66a04528b8ce623fa64ce633196d379f781527c1b1",
        {0: 432, 1: 418},
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, default=SOURCE_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def expected_counts() -> Counter[tuple[str, int]]:
    return Counter(
        {
            (item.source, label): count
            for item in OOD_SOURCES
            for label, count in item.labels.items()
        }
    )


def validate_rows(rows: list[dict[str, Any]]) -> None:
    actual = Counter((str(row["source"]), int(row["ground_truth"])) for row in rows)
    target = expected_counts()
    if actual != target:
        raise ValueError(f"OOD source-label counts drifted: {actual} != {target}")
    identities = [
        (str(row["source"]), int(row["source_row_index"])) for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("OOD rows contain duplicate source row identities")
    for row in rows:
        trajectory = row.get("trajectory")
        if not isinstance(trajectory, str) or not trajectory.strip():
            raise ValueError(
                f"OOD trajectory is empty for source={row.get('source')!r} "
                f"row={row.get('source_row_index')!r}"
            )


def load_source_rows(
    source_directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    filter_log_path = source_directory / Path(FILTER_LOG_FILE).name
    actual_filter_log_sha256 = sha256_file(filter_log_path)
    if actual_filter_log_sha256 != FILTER_LOG_SHA256:
        raise ValueError(
            f"filter-log checksum drift for {filter_log_path}: "
            f"{actual_filter_log_sha256}"
        )
    rows: list[dict[str, Any]] = []
    provenance = []
    for source in OOD_SOURCES:
        path = source_directory / source.filename
        actual_sha256 = sha256_file(path)
        if actual_sha256 != source.sha256:
            raise ValueError(f"source checksum drift for {path}: {actual_sha256}")
        table = pq.read_table(
            path,
            columns=["prompt", "transcript", "ground_truth", "row_index"],
        )
        source_rows = table.to_pylist()
        provenance.append(
            {
                "source": source.source,
                "repo_file": source.repo_file,
                "path": str(path),
                "sha256": source.sha256,
                "rows": len(source_rows),
                "labels": source.labels,
            }
        )
        for source_row_index, source_row in enumerate(source_rows):
            trajectory = source_row["transcript"]
            messages = source_row["prompt"]
            if (
                not isinstance(messages, list)
                or not messages
                or not isinstance(messages[-1], dict)
                or messages[-1].get("content") != trajectory
            ):
                raise ValueError(
                    f"paper prompt/transcript mismatch for {source.source} "
                    f"row {source_row_index}"
                )
            recorded_row_index = source_row["row_index"]
            if recorded_row_index is None:
                raise ValueError(
                    f"paper row index is missing for {source.source} "
                    f"row {source_row_index}"
                )
            rows.append(
                {
                    "source": source.source,
                    "source_file": source.repo_file,
                    "source_row_index": source_row_index,
                    "recorded_row_index": int(recorded_row_index),
                    "ground_truth": int(source_row["ground_truth"]),
                    "trajectory": trajectory,
                }
            )
    validate_rows(rows)
    return rows, provenance


def cost_projection(prompt_tokens: int, rows: int) -> dict[str, float | int]:
    completion_tokens = rows * FIREWORKS_COMPLETION_TOKENS_PER_ROW
    input_cost = prompt_tokens * FIREWORKS_INPUT_USD_PER_MILLION / 1_000_000
    output_cost = completion_tokens * FIREWORKS_OUTPUT_USD_PER_MILLION / 1_000_000
    return {
        "input_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "input_usd": input_cost,
        "output_usd": output_cost,
        "total_usd": input_cost + output_cost,
        "usd_per_1000_rows": (input_cost + output_cost) * 1_000 / rows,
    }


def source_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for source in OOD_SOURCES:
        selected = [row for row in rows if row["source"] == source.source]
        prompt_lengths = [
            int(row["predicted_provider_prompt_tokens"]) for row in selected
        ]
        output[source.source] = {
            "rows": len(selected),
            "labels": dict(
                sorted(Counter(str(row["ground_truth"]) for row in selected).items())
            ),
            "kimi_raw_prompt_tokens": length_summary(
                [int(row["kimi_raw_prompt_tokens"]) for row in selected]
            ),
            "predicted_provider_prompt_tokens": length_summary(prompt_lengths),
            "fireworks_cost_projection": cost_projection(
                sum(prompt_lengths), len(selected)
            ),
        }
    return output


def main() -> None:
    args = parse_args()
    rows, source_provenance = load_source_rows(args.source_directory)
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
        source_identity = (
            f"{SOURCE_REVISION}\0{row['source']}\0{row['source_row_index']}\0"
            f"{row['recorded_row_index']}"
        )
        provider_tokens = (
            int(row["kimi_raw_prompt_tokens"]) + FIREWORKS_CHAT_WRAPPER_TOKENS
        )
        row["predicted_provider_prompt_tokens"] = provider_tokens
        output_rows.append(
            {
                "id": hashlib.sha256(source_identity.encode()).hexdigest()[:24],
                "prompt": prompt,
                "metadata": {
                    "source_repo": SOURCE_REPO,
                    "source_revision": SOURCE_REVISION,
                    "source_file": row["source_file"],
                    "source_license": SOURCE_LICENSE,
                    "source_dataset": row["source"],
                    "source_row_index": row["source_row_index"],
                    "recorded_row_index": row["recorded_row_index"],
                    "ground_truth": row["ground_truth"],
                    "ground_truth_provenance": "paper_author_filtered_eval_artifact",
                    "trajectory_sha256": hashlib.sha256(
                        str(row["trajectory"]).encode("utf-8")
                    ).hexdigest(),
                    "kimi_raw_prompt_tokens": row["kimi_raw_prompt_tokens"],
                    "predicted_provider_prompt_tokens": provider_tokens,
                    "provider_chat_wrapper_tokens": FIREWORKS_CHAT_WRAPPER_TOKENS,
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
    prompt_lengths = [
        int(row["predicted_provider_prompt_tokens"]) for row in rows
    ]
    manifest = {
        "source": {
            "repo": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "license": SOURCE_LICENSE,
            "usage": "internal_research_only; do_not_redistribute",
            "filter_log_file": FILTER_LOG_FILE,
            "filter_log_sha256": FILTER_LOG_SHA256,
            "files": source_provenance,
        },
        "selection": {
            "role": "strict_out_of_distribution_evaluation",
            "rule": "paper_author_filtered_eval_artifacts",
            "rows": len(rows),
            "source_label_counts": {
                source.source: source.labels for source in OOD_SOURCES
            },
            "note": (
                "The prepared inputs contain 1499 Rogue Eval and 850 "
                "SHADE-Arena rows; the released prediction artifact swaps one "
                "row between those counts while retaining 6395 total rows."
            ),
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
            "provider": "Fireworks",
            "provider_chat_wrapper_tokens_per_row": (
                FIREWORKS_CHAT_WRAPPER_TOKENS
            ),
            "provider_wrapper_basis": (
                "exact difference on all 3312 prior successful Fireworks rows"
            ),
            "all": {
                "rows": len(rows),
                "kimi_raw_prompt_tokens": length_summary(
                    [int(row["kimi_raw_prompt_tokens"]) for row in rows]
                ),
                "predicted_provider_prompt_tokens": length_summary(prompt_lengths),
                "fireworks_cost_projection": cost_projection(
                    sum(prompt_lengths), len(rows)
                ),
            },
            "by_source": source_summaries(rows),
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
            "rows": len(output_rows),
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    atomic_write_json(manifest_path, manifest)
    total = manifest["token_accounting"]["all"][
        "predicted_provider_prompt_tokens"
    ]["total"]
    cost = manifest["token_accounting"]["all"]["fireworks_cost_projection"][
        "total_usd"
    ]
    print(
        f"materialized rows={len(output_rows)} provider_prompt_tokens={total} "
        f"fireworks_usd={cost:.6f} output={args.output} "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
