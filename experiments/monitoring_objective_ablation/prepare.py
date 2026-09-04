#!/usr/bin/env python3
"""Materialize rationale-enriched rows and the frozen objective job manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.monitoring_lr_sweep.core import STUDENT_ROWS_SHA256
from experiments.monitoring_objective_ablation.core import (
    ARM_SPECS,
    CAMPAIGN_ID,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    MAX_LENGTH,
    MICRO_BATCH_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    OPTIMIZER_STEPS,
    RANK,
    RAW_SOURCE_SHA256,
    SOFT_TARGETS_SHA256,
    TRAIN_ROWS,
    validate_jobs,
)
from gleipnir.monitoring_systems_screen import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)

ANSWER_PATTERN = re.compile(r"Answer \(0-10\):\s*(\d+)\s*$")
DEFAULT_SOURCE = Path(
    "data/tool_trajectory_monitoring/source/deliberative-monitor-pipeline-2f10f4e/"
    "merged/stride_gloom_crh_s45_o46_ctrl_t7_bal.parquet"
)
DEFAULT_DATA_DIR = Path("data/tool_trajectory_monitoring/distillation_scaling")
DEFAULT_RESULT_DIR = Path("results/monitoring_objective_ablation")


def enrich_student_rows(
    students: list[dict[str, Any]], source_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach the exact selected paper rationale by immutable source row index."""
    if len(students) != TRAIN_ROWS or len(source_rows) != TRAIN_ROWS:
        raise ValueError("training row count drifted")
    enriched = []
    quality_counts: Counter[int] = Counter()
    regenerated_counts: Counter[tuple[int, bool]] = Counter()
    rationale_characters = 0
    for position, student in enumerate(students):
        source_index = int(student["source_row_index"])
        raw = source_rows[source_index]
        messages = raw.get("messages")
        if not isinstance(messages, list) or [
            message.get("role") for message in messages
        ] != ["system", "user", "assistant"]:
            raise ValueError(f"invalid messages at source row {source_index}")
        trajectory = str(messages[1]["content"])
        rationale = str(messages[2]["content"])
        label = int(student["label"])
        answer_match = ANSWER_PATTERN.search(rationale)
        if (
            str(raw["dataset_source"]) != str(student["raw_source"])
            or int(raw["ground_truth"]) != label
            or hashlib.sha256(trajectory.encode()).hexdigest()
            != student["trajectory_sha256"]
            or answer_match is None
            or (int(answer_match.group(1)) >= 5) != bool(label)
        ):
            raise ValueError(f"source/rationale alignment drift at row {position}")
        quality = int(raw["judge_score"])
        if quality not in {7, 8, 9, 10}:
            raise ValueError(f"unexpected rationale quality at row {position}")
        selected = dict(student)
        selected.update(
            student_target=rationale,
            completion_system_prompt=str(messages[0]["content"]),
            completion_prompt=trajectory,
            rationale_sha256=hashlib.sha256(rationale.encode()).hexdigest(),
            rationale_quality_score=quality,
            rationale_regenerated=bool(raw["is_regenerated"]),
            rationale_provenance=(
                "paper-author Gemini-2.5-Pro candidate selected by "
                "Claude-Sonnet-4.5 judge"
            ),
        )
        enriched.append(selected)
        quality_counts[quality] += 1
        regenerated_counts[(label, bool(raw["is_regenerated"]))] += 1
        rationale_characters += len(rationale)
    return enriched, {
        "rows": len(enriched),
        "rationale_characters": rationale_characters,
        "quality_scores": {
            str(score): count for score, count in sorted(quality_counts.items())
        },
        "regenerated_by_label": {
            f"{label}:{str(regenerated).lower()}": count
            for (label, regenerated), count in sorted(regenerated_counts.items())
        },
    }


def make_jobs(
    student_rows: Path, soft_targets: Path, result_dir: Path
) -> list[dict[str, Any]]:
    common = {
        "capacity_kind": "lora",
        "data_scope": "monitoring_only",
        "deception_rows": 0,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": 0,
        "target": "kimi_soft_plus_auxiliary",
        "rank": RANK,
        "lora_alpha": LORA_ALPHA,
        "max_length": MAX_LENGTH,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": 32,
        "gradient_checkpointing": True,
        "selective_torch_compile_backend": "inductor",
        "selective_torch_compile_mode": "default",
        "selective_torch_compile_dynamic": True,
        "trainer_optim": "adamw_torch",
        "optimizer": "adamw",
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "num_train_epochs": 1.0,
        "max_steps": -1,
        "soft_loss_weight": 1.0,
        "direct_loss_weight": 0.0,
        "completion_logits_mode": "selected_positions",
        "completion_projection_chunk_size": 128,
        "mil_temperature": 1.0,
        "mil_top_k": 3,
        "mil_max_instances": 8,
        "require_causal_conv1d": True,
        "require_flash_sdpa": True,
        "fla_disable_backend_dispatch": True,
        "triton_version": "3.7.1",
        "train_rows": TRAIN_ROWS,
        "selection_manifest": None,
        "student_rows": student_rows.as_posix(),
        "soft_targets": soft_targets.as_posix(),
        "student_rows_sha256": sha256_file(student_rows),
        "soft_targets_sha256": sha256_file(soft_targets),
        "learning_rate": LEARNING_RATE,
        "save_steps": OPTIMIZER_STEPS,
    }
    jobs = []
    for spec in ARM_SPECS:
        output = result_dir / "runs" / str(spec["job_name"])
        jobs.append(
            {
                **common,
                **spec,
                "output_dir": output.as_posix(),
                "causal_adapter_dir": (output / "causal_adapter").as_posix(),
                "model_dir": (output / "model").as_posix(),
            }
        )
    validate_jobs(jobs)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    args = parser.parse_args()
    if sha256_file(args.source) != RAW_SOURCE_SHA256:
        raise ValueError("raw rationale source checksum drifted")
    student_path = args.data_dir / "student_rows.jsonl"
    soft_path = args.data_dir / "soft_targets.jsonl"
    if sha256_file(student_path) != STUDENT_ROWS_SHA256:
        raise ValueError("base student rows checksum drifted")
    if sha256_file(soft_path) != SOFT_TARGETS_SHA256:
        raise ValueError("Kimi soft-target checksum drifted")
    import pyarrow.parquet as pq

    source_rows = pq.read_table(args.source).to_pylist()
    enriched, audit = enrich_student_rows(read_jsonl(student_path), source_rows)
    enriched_path = args.data_dir / "rationale_student_rows.jsonl"
    atomic_write_jsonl(enriched_path, enriched)
    preflight_rows = sorted(
        enriched,
        key=lambda row: int(row["student_direct_tokens"]),
        reverse=True,
    )[:32]
    preflight_path = args.result_dir / "selections" / "preflight-longest-32.jsonl"
    atomic_write_jsonl(
        preflight_path,
        [
            {
                "dataset": row["dataset"],
                "index": row["index"],
                "label": row["label"],
                "student_direct_tokens": row["student_direct_tokens"],
            }
            for row in preflight_rows
        ],
    )
    jobs = make_jobs(enriched_path, soft_path, args.result_dir)
    jobs_path = args.result_dir / "jobs.jsonl"
    atomic_write_jsonl(jobs_path, jobs)
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "status": "frozen_before_training",
        "rationale_audit": audit,
        "rationale_student_rows": enriched_path.as_posix(),
        "rationale_student_rows_sha256": sha256_file(enriched_path),
        "kimi_soft_targets_sha256": sha256_file(soft_path),
        "jobs": [job["job_name"] for job in jobs],
        "jobs_sha256": sha256_file(jobs_path),
        "preflight_selection": preflight_path.as_posix(),
        "preflight_selection_sha256": sha256_file(preflight_path),
        "strict_ood_consulted": False,
    }
    atomic_write_json(args.result_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
