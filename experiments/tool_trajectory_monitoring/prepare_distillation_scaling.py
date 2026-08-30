#!/usr/bin/env python3
"""Freeze student rows, Kimi targets, nested subsets, and seven soft-only jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from experiments.tool_trajectory_monitoring.distillation_scaling import (
    CAMPAIGN_SEED,
    EFFECTIVE_BATCH_SIZE,
    LORA_ALPHA,
    LORA_RANK,
    MAX_LENGTH,
    PAPER_SCHEDULE,
    nested_count_selections,
    scaling_job_name,
    validate_jobs,
)
from experiments.tool_trajectory_monitoring.prepare_teacher_training_cache import (
    DEFAULT_SOURCE,
    PAPER_SOURCE_MAP,
    SOURCE_REVISION,
    SOURCE_SHA256,
    extract_trajectory,
)
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from gleipnir.qwen35_adapter_rebase import sha256_file

MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
EXPECTED_REQUEST_HASHES = {
    "Makora": "1fc6c9071634495e6f813d0d2040e8761befb0941ee0b8fa6d31a0deb0a54bdd",
    "Morph": "e5d8b6331b49d69e3a89295a73ccccdcca0405ac8d68b72431b1f6012a76b148",
    "Fireworks": "b60fa90560524f6ae207b1dc89041feb0fa44f29b6d627ac7d3692599b7482c5",
}
DECEPTION_STUDENT_SHA256 = (
    "87cde0d4e6be1edb149a9cd68d4dcf2fbef8c8e8e28487d96569246b5a935fe7"
)
DECEPTION_SOFT_SHA256 = (
    "2100781d05a2f61328568524233d2a6be583a95b10c28bb6b265e80abd64d273"
)
DEFAULT_PROMPTS = Path("data/tool_trajectory_monitoring/teacher_training/prompts.jsonl")
DEFAULT_SCORES = (
    Path("results/tool_trajectory_monitoring/kimi_k3_training_logits.jsonl"),
    Path("results/tool_trajectory_monitoring/kimi_k3_training_logits_morph.jsonl"),
    Path("results/tool_trajectory_monitoring/kimi_k3_training_logits_fireworks.jsonl"),
)
DEFAULT_DECEPTION_STUDENTS = Path("data/data_scaling/student_rows.jsonl")
DEFAULT_DECEPTION_SOFT = Path("data/data_scaling/soft_targets.jsonl")
DEFAULT_DATA_DIR = Path("data/tool_trajectory_monitoring/distillation_scaling")
DEFAULT_RESULT_DIR = Path("results/tool_trajectory_distillation_scaling")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def load_prompt_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = read_jsonl(path)
    by_source_index = {
        int(row["metadata"]["source_row_index"]): row for row in rows
    }
    if len(rows) != 8_688 or len(by_source_index) != len(rows):
        raise ValueError("teacher prompt artifact must contain 8,688 unique rows")
    return by_source_index


def materialize_student_rows(
    source: Path,
    prompt_rows: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    if sha256_file(source) != SOURCE_SHA256:
        raise ValueError(f"training source checksum drifted: {source}")
    prompt_set = load_prompt_set()
    source_rows = pq.read_table(
        source,
        columns=[
            "messages",
            "prompt_id",
            "ground_truth",
            "dataset_source",
            "sample_index",
            "original_prompt_id",
        ],
    ).to_pylist()
    output = []
    for source_row_index, source_row in enumerate(source_rows):
        cached = prompt_rows[source_row_index]
        metadata = cached["metadata"]
        trajectory = extract_trajectory(source_row["messages"])
        trajectory_sha256 = hashlib.sha256(trajectory.encode()).hexdigest()
        if trajectory_sha256 != metadata["trajectory_sha256"]:
            raise ValueError(f"trajectory checksum drift at row {source_row_index}")
        label = int(source_row["ground_truth"])
        raw_source = str(source_row["dataset_source"])
        source_dataset = PAPER_SOURCE_MAP[raw_source]
        if label != int(metadata["ground_truth"]) or source_dataset != str(
            metadata["source_dataset"]
        ):
            raise ValueError(f"teacher metadata drift at row {source_row_index}")
        student_prompt = prompt_set.student.render(trajectory)
        output.append(
            {
                "dataset": f"tool_trajectory/{source_dataset}",
                "index": str(cached["id"]),
                "label": label,
                "label_match": True,
                "parse_error": False,
                "student_prompt": student_prompt,
                "student_prompt_sha256": hashlib.sha256(
                    student_prompt.encode()
                ).hexdigest(),
                "student_target": prompt_set.student.completion(str(label)),
                "source": "kimi_k3_binary_logit_cache",
                "source_dataset": source_dataset,
                "raw_source": raw_source,
                "source_revision": SOURCE_REVISION,
                "source_row_index": source_row_index,
                "prompt_id": int(source_row["prompt_id"]),
                "original_prompt_id": int(source_row["original_prompt_id"]),
                "sample_index": int(source_row["sample_index"]),
                "lineage_group": f"{raw_source}:{source_row['original_prompt_id']}",
                "trajectory_sha256": trajectory_sha256,
                "paper_input_tokens": int(metadata["paper_input_tokens"]),
                "prompt_set_id": prompt_set.prompt_set_id,
                "student_template_sha256": prompt_set.student.template_sha256,
                "teacher_template_sha256": metadata["prompt_template_sha256"],
                "teacher_rendered_prompt_sha256": metadata[
                    "rendered_prompt_sha256"
                ],
            }
        )
    if len(output) != 8_688:
        raise ValueError(f"expected 8,688 training rows, got {len(output)}")
    return output


def load_soft_targets(
    score_paths: list[Path],
    students: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    score_by_id: dict[str, dict[str, Any]] = {}
    provider_counts: Counter[str] = Counter()
    for path in score_paths:
        for row in read_jsonl(path):
            identity = str(row["id"])
            if identity in score_by_id:
                raise ValueError(f"duplicate Kimi score ID {identity!r}")
            provider = str(row["provider"])
            expected_hash = EXPECTED_REQUEST_HASHES.get(provider)
            if expected_hash is None or row["request_settings_sha256"] != expected_hash:
                raise ValueError(f"request identity drift for {identity!r}")
            if row["prompt_sha256"] != row["metadata"]["rendered_prompt_sha256"]:
                raise ValueError(f"rendered prompt drift for {identity!r}")
            score_by_id[identity] = row
            provider_counts[provider] += 1
    student_ids = {str(row["index"]) for row in students}
    if set(score_by_id) != student_ids or len(score_by_id) != 8_688:
        raise ValueError("provider score union does not exactly cover student rows")

    targets = []
    for student in students:
        identity = str(student["index"])
        score = score_by_id[identity]
        label = int(score["metadata"]["ground_truth"])
        probability = float(score["score"])
        if label != int(student["label"]) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"invalid soft target for {identity!r}")
        targets.append(
            {
                "dataset": student["dataset"],
                "index": identity,
                "label": label,
                "soft_target": probability,
                "target_probs": score["target_probs"],
                "target_logprobs": score["target_logprobs"],
                "teacher_model": score["model"],
                "provider": score["provider"],
                "request_settings_sha256": score["request_settings_sha256"],
                "rendered_prompt_sha256": score["prompt_sha256"],
                "received_at": score["received_at"],
            }
        )
    if provider_counts != Counter({"Makora": 5_071, "Morph": 305, "Fireworks": 3_312}):
        raise ValueError(f"provider counts drifted: {provider_counts}")
    return targets


def add_student_token_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    lengths = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["student_prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        direct_ids = tokenizer.encode(prompt + "Prediction:", add_special_tokens=False)
        length = len(direct_ids)
        row["student_direct_tokens"] = length
        lengths.append(length)
    maximum = max(lengths)
    if maximum > MAX_LENGTH:
        raise ValueError(
            f"student direct context exceeds max_length: {maximum} > {MAX_LENGTH}"
        )
    ordered = sorted(lengths)
    return {
        "tokenizer": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "mean": sum(lengths) / len(lengths),
        "p50": ordered[len(ordered) // 2],
        "p90": ordered[math.floor(0.90 * (len(ordered) - 1))],
        "p95": ordered[math.floor(0.95 * (len(ordered) - 1))],
        "p99": ordered[math.floor(0.99 * (len(ordered) - 1))],
        "max": maximum,
        "configured_max_length": MAX_LENGTH,
        "truncated_rows": 0,
    }


def make_jobs(data_dir: Path, result_dir: Path) -> list[dict[str, Any]]:
    common = {
        "seed": CAMPAIGN_SEED,
        "target": "kimi_soft",
        "rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "max_length": MAX_LENGTH,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": EFFECTIVE_BATCH_SIZE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "learning_rate": 5e-5,
        "optimizer": "adamw",
        "soft_loss_weight": 1.0,
        "direct_loss_weight": 0.0,
    }
    jobs = []
    for rows, steps in PAPER_SCHEDULE.items():
        job_name = scaling_job_name(rows)
        jobs.append(
            {
                **common,
                "job_name": job_name,
                "design_role": "paper_count_kimi_soft_curve",
                "train_rows": rows,
                "max_steps": steps,
                "num_train_epochs": 1.0,
                "student_rows": str(data_dir / "student_rows.jsonl"),
                "soft_targets": str(data_dir / "soft_targets.jsonl"),
                "selection_manifest": str(
                    result_dir / "selections" / f"n{rows:05d}-seed0.jsonl"
                ),
                "save_steps": max(1, steps // 3),
            }
        )
    mixed_rows = 8_688 + 13_149
    jobs.append(
        {
            **common,
            "job_name": f"soft-n{mixed_rows}-mixed-seed{CAMPAIGN_SEED}",
            "design_role": "full_tool_plus_prior_deception_transfer",
            "train_rows": mixed_rows,
            "max_steps": -1,
            "num_train_epochs": 3.0,
            "student_rows": str(data_dir / "mixed_student_rows.jsonl"),
            "soft_targets": str(data_dir / "mixed_soft_targets.jsonl"),
            "selection_manifest": None,
            "save_steps": math.ceil(mixed_rows / EFFECTIVE_BATCH_SIZE),
        }
    )
    validate_jobs(jobs)
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--scores", type=Path, action="append")
    parser.add_argument(
        "--deception-students", type=Path, default=DEFAULT_DECEPTION_STUDENTS
    )
    parser.add_argument("--deception-soft", type=Path, default=DEFAULT_DECEPTION_SOFT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--skip-token-audit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_paths = args.scores or list(DEFAULT_SCORES)
    prompt_rows = load_prompt_rows(args.prompts)
    students = materialize_student_rows(args.source, prompt_rows)
    token_summary = (
        {"status": "skipped"}
        if args.skip_token_audit
        else add_student_token_counts(students)
    )
    soft_targets = load_soft_targets(score_paths, students)

    if sha256_file(args.deception_students) != DECEPTION_STUDENT_SHA256:
        raise ValueError("prior-deception student artifact checksum drifted")
    if sha256_file(args.deception_soft) != DECEPTION_SOFT_SHA256:
        raise ValueError("prior-deception soft-target artifact checksum drifted")
    deception_students = read_jsonl(args.deception_students)
    deception_soft = read_jsonl(args.deception_soft)
    if len(deception_students) != 13_149 or len(deception_soft) != 13_149:
        raise ValueError("prior-deception artifacts must contain 13,149 rows")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    student_path = args.data_dir / "student_rows.jsonl"
    soft_path = args.data_dir / "soft_targets.jsonl"
    mixed_student_path = args.data_dir / "mixed_student_rows.jsonl"
    mixed_soft_path = args.data_dir / "mixed_soft_targets.jsonl"
    atomic_write_jsonl(student_path, students)
    atomic_write_jsonl(soft_path, soft_targets)
    atomic_write_jsonl(mixed_student_path, students + deception_students)
    atomic_write_jsonl(mixed_soft_path, soft_targets + deception_soft)

    selections = nested_count_selections(students, PAPER_SCHEDULE, CAMPAIGN_SEED)
    selection_summary = {}
    for count, selected in selections.items():
        path = args.result_dir / "selections" / f"n{count:05d}-seed0.jsonl"
        rows = [
            {
                "dataset": row["dataset"],
                "index": row["index"],
                "label": row["label"],
            }
            for row in selected
        ]
        atomic_write_jsonl(path, rows)
        selection_summary[str(count)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "source_label_counts": {
                f"{source}|{label}": value
                for (source, label), value in sorted(
                    Counter(
                        (str(row["source_dataset"]), int(row["label"]))
                        for row in selected
                    ).items()
                )
            },
        }

    if not all("student_direct_tokens" in row for row in students):
        raise ValueError("token audit is required before freezing the memory preflight")
    longest = sorted(
        students,
        key=lambda row: (-int(row["student_direct_tokens"]), str(row["index"])),
    )[:EFFECTIVE_BATCH_SIZE]
    preflight_path = args.result_dir / "selections" / "preflight-longest-32.jsonl"
    atomic_write_jsonl(
        preflight_path,
        [
            {
                "dataset": row["dataset"],
                "index": row["index"],
                "label": row["label"],
            }
            for row in longest
        ],
    )

    jobs = make_jobs(args.data_dir, args.result_dir)
    artifact_checksums = {
        str(student_path): sha256_file(student_path),
        str(soft_path): sha256_file(soft_path),
        str(mixed_student_path): sha256_file(mixed_student_path),
        str(mixed_soft_path): sha256_file(mixed_soft_path),
    }
    for job in jobs:
        job["student_rows_sha256"] = artifact_checksums[str(job["student_rows"])]
        job["soft_targets_sha256"] = artifact_checksums[str(job["soft_targets"])]
        if job["selection_manifest"] is not None:
            job["selection_sha256"] = sha256_file(Path(job["selection_manifest"]))
    jobs_path = args.result_dir / "jobs.jsonl"
    atomic_write_jsonl(jobs_path, jobs)
    manifest = {
        "campaign_id": "tool-trajectory-kimi-soft-scaling-r128-v1",
        "hypothesis": (
            "Kimi K3 soft targets yield a useful Qwen3.5-9B scaling curve, and "
            "adding prior deception soft-distillation rows may improve transfer."
        ),
        "arms": "soft_distillation_only; no hard-label adapters",
        "seed": CAMPAIGN_SEED,
        "paper_schedule": PAPER_SCHEDULE,
        "student_model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "recipe": {
            "rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "learning_rate": 5e-5,
            "optimizer": "adamw",
            "target": "Kimi K3 normalized literal-1 probability",
            "loss": "binary soft BCE at selected literal 0|1 logits",
            "quantization": "NF4 double-quantized QLoRA with BF16 compute",
            "micro_batch_size": 1,
            "gradient_accumulation_steps": EFFECTIVE_BATCH_SIZE,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "mixed_epochs": 3,
        },
        "token_summary": token_summary,
        "artifacts": {
            "student_rows": {
                "path": str(student_path),
                "rows": len(students),
                "sha256": sha256_file(student_path),
            },
            "soft_targets": {
                "path": str(soft_path),
                "rows": len(soft_targets),
                "sha256": sha256_file(soft_path),
            },
            "mixed_student_rows": {
                "path": str(mixed_student_path),
                "rows": len(students) + len(deception_students),
                "sha256": sha256_file(mixed_student_path),
                "prior_deception_sha256": DECEPTION_STUDENT_SHA256,
            },
            "mixed_soft_targets": {
                "path": str(mixed_soft_path),
                "rows": len(soft_targets) + len(deception_soft),
                "sha256": sha256_file(mixed_soft_path),
                "prior_deception_sha256": DECEPTION_SOFT_SHA256,
            },
            "teacher_score_shards": {
                str(path): sha256_file(path) for path in score_paths
            },
        },
        "selection": selection_summary,
        "preflight": {
            "selection_manifest": str(preflight_path),
            "selection_sha256": sha256_file(preflight_path),
            "rows": len(longest),
            "minimum_tokens": min(int(row["student_direct_tokens"]) for row in longest),
            "maximum_tokens": max(int(row["student_direct_tokens"]) for row in longest),
            "optimizer_steps": 1,
        },
        "jobs": {"path": str(jobs_path), "count": len(jobs)},
        "stop_conditions": [
            "any input, prompt, request, or selection checksum drifts",
            "any selection is non-nested or arms other than Kimi soft appear",
            "FLA 0.5.2 is unavailable or gated-delta kernels fall back",
            "the rank-128 longest-sequence preflight OOMs",
            "training changes NF4, selected-position logits, or effective batch 32",
        ],
    }
    atomic_write_json(args.result_dir / "manifest.json", manifest)
    print(
        f"prepared {len(jobs)} soft-only jobs; tool_rows={len(students)} "
        f"mixed_rows={len(students) + len(deception_students)} "
        f"max_student_tokens={token_summary.get('max')}"
    )


if __name__ == "__main__":
    main()
