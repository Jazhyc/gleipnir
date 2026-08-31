"""Materialize a balanced, prompt-aware teacher canary from pinned sources."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from experiments.tool_trajectory_monitoring.teacher_canary import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_SEED,
    SOURCE_FILES,
    SOURCE_LICENSE,
    SOURCE_REPO,
    SOURCE_REVISION,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
    atomic_write_json,
    atomic_write_jsonl,
    load_source_candidates,
    select_balanced_candidates,
    sha256_file,
)

DEFAULT_SOURCE_DIRECTORY = Path(
    "data/tool_trajectory_monitoring/source/blackboxmonitorsMATS-ccb77e3"
)
DEFAULT_OUTPUT = Path("data/tool_trajectory_monitoring/canary/prompts.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-directory",
        type=Path,
        default=DEFAULT_SOURCE_DIRECTORY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
    )
    parser.add_argument("--tokenizer", default=TOKENIZER_ID)
    parser.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
    return parser.parse_args()


def add_token_counts(
    candidates: list[dict[str, Any]],
    tokenizer: Any,
    *,
    batch_size: int = 16,
) -> None:
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        encoded = tokenizer(
            [row["formatted_transcript"] for row in batch],
            add_special_tokens=False,
            return_length=True,
        )
        for row, length in zip(batch, encoded["length"], strict=True):
            row["input_tokens"] = int(length)


def main() -> None:
    args = parse_args()
    if args.max_input_tokens < 1:
        raise ValueError("max input tokens must be positive")
    candidates = load_source_candidates(args.source_directory)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
    )
    add_token_counts(candidates, tokenizer)
    eligible = [
        row for row in candidates if row["input_tokens"] <= args.max_input_tokens
    ]
    sources = sorted({str(row["source_dataset"]) for row in candidates})
    selected = select_balanced_candidates(
        eligible,
        total_rows=args.rows,
        seed=args.seed,
        sources=sources,
    )

    prompt_set = load_prompt_set()
    template = prompt_set.teacher
    cache_prefix_sha256 = hashlib.sha256(
        template.cache_prefix.encode("utf-8")
    ).hexdigest()
    output_rows = []
    for row in selected:
        trajectory = str(row["formatted_transcript"])
        prompt = template.render(trajectory)
        source_id = str(row["id"])
        opaque_id = hashlib.sha256(
            f"{SOURCE_REVISION}\0{row['source_dataset']}\0{source_id}".encode()
        ).hexdigest()[:24]
        output_rows.append(
            {
                "id": opaque_id,
                "prompt": prompt,
                "metadata": {
                    "source_repo": SOURCE_REPO,
                    "source_revision": SOURCE_REVISION,
                    "source_license": SOURCE_LICENSE,
                    "source_file": row["source_file"],
                    "source_dataset": row["source_dataset"],
                    "source_row_index": row["source_row_index"],
                    "source_id": source_id,
                    "ground_truth": int(row["ground_truth"]),
                    "ground_truth_provenance": "source_dataset",
                    "trajectory_sha256": hashlib.sha256(
                        trajectory.encode("utf-8")
                    ).hexdigest(),
                    "input_tokens": row["input_tokens"],
                    "tokenizer_id": args.tokenizer,
                    "tokenizer_revision": args.tokenizer_revision,
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
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest = {
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_license": SOURCE_LICENSE,
        "source_files": {
            filename: {
                "sha256": sha256_file(args.source_directory / filename),
                "bytes": (args.source_directory / filename).stat().st_size,
            }
            for filename in SOURCE_FILES
        },
        "selection": {
            "seed": args.seed,
            "requested_rows": args.rows,
            "max_input_tokens": args.max_input_tokens,
            "candidate_rows": len(candidates),
            "eligible_rows": len(eligible),
            "selected_ids_in_request_order": [row["id"] for row in output_rows],
        },
        "tokenizer": {
            "id": args.tokenizer,
            "revision": args.tokenizer_revision,
        },
        "prompt": {
            "prompt_set_id": prompt_set.prompt_set_id,
            "template_sha256": template.template_sha256,
            "cache_prefix_sha256": cache_prefix_sha256,
            "cache_prefix_chars": len(template.cache_prefix),
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
            "rows": len(output_rows),
        },
    }
    atomic_write_json(manifest_path, manifest)
    print(
        f"materialized rows={len(output_rows)} eligible={len(eligible)}/"
        f"{len(candidates)} output={args.output} manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
