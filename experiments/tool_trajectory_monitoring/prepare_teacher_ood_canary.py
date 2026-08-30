#!/usr/bin/env python3
"""Select a deterministic balanced canary from frozen Kimi OOD prompts."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.tool_trajectory_monitoring.prepare_teacher_ood_cache import (
    DEFAULT_OUTPUT as DEFAULT_FULL_INPUT,
)
from experiments.tool_trajectory_monitoring.prepare_teacher_ood_cache import (
    FIREWORKS_COMPLETION_TOKENS_PER_ROW,
    FIREWORKS_INPUT_USD_PER_MILLION,
    FIREWORKS_OUTPUT_USD_PER_MILLION,
    OOD_SOURCES,
    cost_projection,
)
from experiments.tool_trajectory_monitoring.teacher_canary import (
    DEFAULT_SEED,
    atomic_write_json,
    atomic_write_jsonl,
    load_jsonl,
    sha256_file,
    stable_sample_key,
)

DEFAULT_OUTPUT = Path(
    "data/tool_trajectory_monitoring/teacher_ood_canary/prompts.jsonl"
)
DEFAULT_ROWS_PER_STRATUM = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_FULL_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--rows-per-stratum", type=int, default=DEFAULT_ROWS_PER_STRATUM
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def select_balanced_ood_rows(
    rows: list[dict[str, Any]],
    *,
    rows_per_stratum: int,
    seed: int,
) -> list[dict[str, Any]]:
    if rows_per_stratum < 1:
        raise ValueError("rows_per_stratum must be positive")
    selected: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for source in OOD_SOURCES:
        for label in (0, 1):
            eligible = [
                row
                for row in rows
                if row.get("metadata", {}).get("source_dataset") == source.source
                and int(row.get("metadata", {}).get("ground_truth")) == label
            ]
            eligible.sort(
                key=lambda row: stable_sample_key(
                    seed,
                    source.source,
                    label,
                    str(row["id"]),
                )
            )
            if len(eligible) < rows_per_stratum:
                raise ValueError(
                    f"stratum source={source.source!r} label={label} has "
                    f"{len(eligible)} rows; need {rows_per_stratum}"
                )
            selected[(source.source, label)] = eligible[:rows_per_stratum]

    interleaved = []
    for index in range(rows_per_stratum):
        for source in OOD_SOURCES:
            for label in (0, 1):
                interleaved.append(selected[(source.source, label)][index])
    return interleaved


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    selected = select_balanced_ood_rows(
        rows,
        rows_per_stratum=args.rows_per_stratum,
        seed=args.seed,
    )
    ids = [str(row["id"]) for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("selected OOD canary rows contain duplicate IDs")
    atomic_write_jsonl(args.output, selected)

    source_label_counts = Counter(
        (
            str(row["metadata"]["source_dataset"]),
            str(row["metadata"]["ground_truth"]),
        )
        for row in selected
    )
    provider_prompt_tokens = sum(
        int(row["metadata"]["predicted_provider_prompt_tokens"])
        for row in selected
    )
    manifest = {
        "input": {
            "path": str(args.input),
            "sha256": sha256_file(args.input),
            "rows": len(rows),
        },
        "selection": {
            "rule": "stable_sha256_prefix_per_source_label_stratum",
            "seed": args.seed,
            "rows_per_stratum": args.rows_per_stratum,
            "rows": len(selected),
            "source_label_counts": {
                f"{source}:{label}": count
                for (source, label), count in sorted(source_label_counts.items())
            },
            "ordered_ids_sha256": hashlib.sha256(
                "\n".join(ids).encode("utf-8")
            ).hexdigest(),
            "usage": (
                "OOD interface and quality canary; do not tune prompts, "
                "thresholds, or model settings from these labels"
            ),
        },
        "fireworks_cost_projection": cost_projection(
            provider_prompt_tokens, len(selected)
        ),
        "rates": {
            "input_usd_per_million": FIREWORKS_INPUT_USD_PER_MILLION,
            "output_usd_per_million": FIREWORKS_OUTPUT_USD_PER_MILLION,
            "completion_tokens_per_row": FIREWORKS_COMPLETION_TOKENS_PER_ROW,
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
            "rows": len(selected),
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    atomic_write_json(manifest_path, manifest)
    print(
        f"selected rows={len(selected)} provider_prompt_tokens="
        f"{provider_prompt_tokens} projected_fireworks_usd="
        f"{manifest['fireworks_cost_projection']['total_usd']:.6f} "
        f"output={args.output} manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
