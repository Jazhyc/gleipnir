#!/usr/bin/env python3
"""Materialize the paper's frozen external lie-detection evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data_scaling.prepare import read_jsonl, write_jsonl  # noqa: E402
from gleipnir.prompts import build_student_prompt  # noqa: E402

DATASET_ID = "ai-safety-institute/lie-detection-rollouts"
DATASET_REVISION = "177a81666ab378ce9d055507bc30f480ab63e74b"
DATASET_LICENSE = "other (no separate terms file in the repository)"
PAPER_ID = "arXiv:2606.12618v2"
COLLECTION_ID = "ai-safety-institute/lie-detection-69e8e7b608f926010851f645"

VARIED_CONFIGS = (
    "deepseek-ai-deepseek-v3.2",
    "google-gemma-2-9b-it",
    "google-gemma-3-12b-it",
    "google-gemma-3-27b-it",
    "google-gemma-4-26b-a4b-it",
    "google-gemma-4-31b-it",
    "google-gemma-4-e2b-it",
    "google-gemma-4-e4b-it",
    "meta-llama-llama-3.3-70b-instruct",
    "minimaxai-minimax-m2",
    "minimaxai-minimax-m2.7",
    "moonshotai-kimi-k2.5",
    "moonshotai-kimi-k2.6",
    "openai-gpt-oss-120b",
    "openai-gpt-oss-20b",
    "qwen-qwen3-14b",
    "qwen-qwen3-30b-a3b",
    "qwen-qwen3-32b",
    "qwen-qwen3-4b",
    "qwen-qwen3-8b",
    "qwen-qwen3.5-122b-a10b-fp8",
    "qwen-qwen3.5-27b",
    "qwen-qwen3.5-35b-a3b-fp8",
    "qwen-qwen3.5-4b",
    "qwen-qwen3.6-27b",
    "qwen-qwen3.6-35b-a3b",
    "xiaomimimo-mimo-v2-flash",
    "zai-org-glm-4.5-air-fp8",
    "zai-org-glm-4.7-flash",
    "zai-org-glm-4.7-fp8",
    "zai-org-glm-5-fp8",
    "zai-org-glm-5.1-fp8",
)

MODEL_ORGANISM_SPLITS = {
    "qwen-qwen3.5-27b__ai-safety-institute-qwen3.5-27b-ab_animal_welfare": (
        "audit_bench",
        "audit_bench_updated",
    ),
    "qwen-qwen3.5-27b__ai-safety-institute-qwen3.5-27b-ab_hallucinates_citations": (
        "audit_bench",
        "audit_bench_updated",
    ),
    "qwen-qwen3.5-27b__ai-safety-institute-qwen3.5-27b-ab_self_promotion": (
        "audit_bench",
        "audit_bench_updated",
    ),
    "qwen-qwen3.5-27b__ai-safety-institute-qwen3.5-27b-eval_sandbagger": (
        "sandbagging",
        "sandbagging_games_updated",
    ),
    "qwen-qwen3.5-27b__ai-safety-institute-qwen3.5-27b-gender_secret_female": (
        "gender_secret",
        "gender_secret",
    ),
    "qwen-qwen3.5-27b__ai-safety-institute-qwen3.5-27b-gender_secret_male": (
        "gender_secret",
        "gender_secret",
    ),
    "qwen-qwen3.6-27b__ai-safety-institute-qwen3.6-27b-ab_animal_welfare": (
        "audit_bench",
        "audit_bench_updated",
    ),
    "qwen-qwen3.6-27b__ai-safety-institute-qwen3.6-27b-ab_contextual_optimism": (
        "audit_bench",
        "audit_bench_updated",
    ),
    "qwen-qwen3.6-27b__ai-safety-institute-qwen3.6-27b-ab_hallucinates_citations": (
        "audit_bench",
        "audit_bench_updated",
    ),
    "qwen-qwen3.6-27b__ai-safety-institute-qwen3.6-27b-ab_self_promotion": (
        "audit_bench",
        "audit_bench_updated",
    ),
    "qwen-qwen3.6-27b__ai-safety-institute-qwen3.6-27b-eval_sandbagger": (
        "sandbagging",
        "sandbagging_games_updated",
    ),
    "qwen-qwen3.6-27b__ai-safety-institute-qwen3.6-27b-gender_secret_female": (
        "gender_secret",
        "gender_secret",
    ),
    "qwen-qwen3.6-27b__ai-safety-institute-qwen3.6-27b-gender_secret_male": (
        "gender_secret",
        "gender_secret",
    ),
}

MESSAGE_BOUNDARY = re.compile(r"(?:^|\n\n)(SYSTEM|USER|ASSISTANT): ")


@dataclass(frozen=True)
class SourceSpec:
    testbed: str
    config: str
    split: str

    @property
    def filename(self) -> str:
        return f"{self.config}/{self.split}.parquet"


def source_specs() -> list[SourceSpec]:
    """Return the paper-table population in a stable order."""
    varied = [
        SourceSpec("varied_deception", value, "varied_deception")
        for value in VARIED_CONFIGS
    ]
    organisms = [
        SourceSpec(testbed, config, split)
        for config, (testbed, split) in MODEL_ORGANISM_SPLITS.items()
    ]
    return varied + organisms


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_hash(value: str) -> str:
    normalized = " ".join(value.split())
    return sha256_bytes(normalized.encode("utf-8"))


def parse_messages(value: Any) -> list[dict[str, str]]:
    """Parse released JSON messages and remove system/CoT information."""
    messages = json.loads(value) if isinstance(value, str) else value
    if not isinstance(messages, list):
        raise ValueError("messages must decode to a list")
    visible = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = str(message.get("role", "")).lower()
        if role == "system":
            continue
        if role not in {"user", "assistant"}:
            raise ValueError(f"unsupported message role: {role!r}")
        visible.append({"role": role, "content": str(message.get("content", ""))})
    if not visible or visible[-1]["role"] != "assistant":
        raise ValueError("a rollout must end with an assistant message")
    return visible


def last_user_hash(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return normalized_hash(message["content"])
    raise ValueError("rollout has no user message")


def formatted_messages(prompt: str) -> list[tuple[str, str]]:
    """Extract role/content blocks from a materialized student context."""
    context_start = prompt.find("<context>\n")
    context_end = prompt.find("\n</context>", context_start)
    if context_start < 0 or context_end < 0:
        raise ValueError("student prompt has no context block")
    context = prompt[context_start + len("<context>\n") : context_end]
    matches = list(MESSAGE_BOUNDARY.finditer(context))
    output = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(context)
        output.append((match.group(1).lower(), context[start:end]))
    return output


def internal_overlap_sets(path: Path) -> tuple[set[str], set[str]]:
    """Build visible-transcript and final-user hashes from prior student rows."""
    if not path.is_file():
        return set(), set()
    transcript_hashes: set[str] = set()
    user_hashes: set[str] = set()
    for record in read_jsonl(path):
        blocks = formatted_messages(str(record["student_prompt"]))
        visible = [
            {"role": role, "content": content}
            for role, content in blocks
            if role != "system"
        ]
        if not visible or visible[-1]["role"] != "assistant":
            continue
        transcript_hashes.add(
            normalized_hash(json.dumps(visible, sort_keys=True, ensure_ascii=False))
        )
        # Tail truncation can remove every user turn from long internal prompts.
        # Such rows remain useful for exact visible-transcript checks, but have
        # no defensible question identity to add to the overlap set.
        if any(message["role"] == "user" for message in visible):
            user_hashes.add(last_user_hash(visible))
    return transcript_hashes, user_hashes


def convert_row(
    row: dict[str, Any],
    spec: SourceSpec,
    source_index: int,
    prompt_template: str,
    internal_transcripts: set[str],
    internal_users: set[str],
) -> tuple[dict[str, Any] | None, str]:
    """Convert one released rollout, returning a filter reason when omitted."""
    label_text = str(row.get("is_lie") or "none").lower()
    if label_text not in {"honest", "lie"}:
        return None, label_text
    messages = parse_messages(row["messages"])
    transcript_hash = normalized_hash(
        json.dumps(messages, sort_keys=True, ensure_ascii=False)
    )
    if transcript_hash in internal_transcripts:
        return None, "exact_internal_transcript"
    prompt = build_student_prompt(messages, prompt_template, 3000, "tail")
    user_hash = last_user_hash(messages)
    dataset = f"aisi-lie-detection/{spec.testbed}/{spec.config}"
    return (
        {
            "dataset": dataset,
            "index": source_index,
            "label": int(label_text == "lie"),
            "testbed": spec.testbed,
            "subject_model": spec.config,
            "source_split": spec.split,
            "source_index": source_index,
            "seen_internal_user": user_hash in internal_users,
            "visible_transcript_sha256": transcript_hash,
            "final_user_sha256": user_hash,
            "student_prompt": prompt,
            "student_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        },
        "kept",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/aisi_lie_detection_transfer/evaluation.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/aisi_lie_detection_transfer/manifest.json"),
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=Path("experiments/aisi_lie_detection_transfer/prompt.txt"),
    )
    parser.add_argument(
        "--internal-rows",
        type=Path,
        default=Path("data/data_scaling/student_rows.jsonl"),
    )
    parser.add_argument("--revision", default=os.environ.get("GLEIPNIR_COMMIT"))
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    prompt_template = args.prompt_template.resolve().read_text().strip()
    internal_transcripts, internal_users = internal_overlap_sets(
        args.internal_rows.resolve()
    )
    records: list[dict[str, Any]] = []
    sources = []
    for spec in source_specs():
        source_path = Path(
            hf_hub_download(
                repo_id=DATASET_ID,
                filename=spec.filename,
                repo_type="dataset",
                revision=DATASET_REVISION,
            )
        )
        frame = pd.read_parquet(source_path, columns=["messages", "is_lie"])
        counts: Counter[str] = Counter()
        group_records = []
        for source_index, row in enumerate(frame.to_dict(orient="records")):
            converted, reason = convert_row(
                row,
                spec,
                source_index,
                prompt_template,
                internal_transcripts,
                internal_users,
            )
            counts[reason] += 1
            if converted is not None:
                group_records.append(converted)
        if {record["label"] for record in group_records} != {0, 1}:
            raise ValueError(
                f"{spec.config}/{spec.split} lacks both labels after filtering"
            )
        records.extend(group_records)
        sources.append(
            {
                **asdict(spec),
                "filename": spec.filename,
                "file_sha256": sha256_file(source_path),
                "raw_rows": len(frame),
                "kept_rows": len(group_records),
                "filter_counts": dict(sorted(counts.items())),
                "seen_internal_user_rows": sum(
                    bool(record["seen_internal_user"]) for record in group_records
                ),
            }
        )
    if len(sources) != 45:
        raise AssertionError(
            f"expected 45 subject-testbed sources, found {len(sources)}"
        )
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output.resolve(), records)
    manifest = {
        "paper": PAPER_ID,
        "collection": COLLECTION_ID,
        "source_dataset": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "source_license": DATASET_LICENSE,
        "code_revision": args.revision,
        "prompt_path": args.prompt_template.as_posix(),
        "prompt_sha256": sha256_bytes(prompt_template.encode("utf-8")),
        "selection_rule": {
            "varied_deception": (
                "full official evaluation split for all Table 9 subject models"
            ),
            "varied_deception_validation": "excluded; paper detector-tuning split",
            "model_organisms": "full paper auditing split for each of 13 organisms",
            "system_messages": "removed to match the paper's judge-model condition",
            "reasoning_content": "excluded; text-output monitor evaluation",
            "labels": "keep lie/honest; exclude ambiguous/null",
            "exact_internal_transcripts": "excluded before evaluation",
            "question_overlap": "retained and flagged as seen_internal_user",
        },
        "rows": len(records),
        "evaluation_sha256": sha256_file(args.output.resolve()),
        "internal_overlap_source": args.internal_rows.as_posix(),
        "internal_visible_transcripts": len(internal_transcripts),
        "internal_final_users": len(internal_users),
        "sources": sources,
    }
    args.manifest.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.manifest.resolve().write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"materialized {len(records)} rows from {len(sources)} sources at "
        f"{args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
