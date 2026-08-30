#!/usr/bin/env python3
"""Freeze the one-job Qwen3.5-4B mixed-data soft-distillation follow-up."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from experiments.tool_trajectory_monitoring.distillation_scaling import (
    CAMPAIGN_SEED,
    EFFECTIVE_BATCH_SIZE,
    LORA_ALPHA,
    LORA_RANK,
    MAX_LENGTH,
)
from experiments.tool_trajectory_monitoring.prepare_distillation_scaling import (
    atomic_write_json,
    atomic_write_jsonl,
)
from gleipnir.qwen35_adapter_rebase import sha256_file

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
JOB_NAME = "soft-n21837-mixed-qwen35-4b-seed0"
MIXED_ROWS = 21_837
MIXED_STUDENT_SHA256 = (
    "f9b0a322f277991edae2510b658a2298c0c585d402b39f36d6ac86352bf1a76b"
)
MIXED_SOFT_SHA256 = (
    "50f163f88b42ff8b6fc04d5ff1334766d6326895ad89c87501a93eeaeae137ee"
)
TOKENIZER_JSON_SHA256 = (
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)
TOKENIZER_CONFIG_SHA256 = (
    "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
)
DEFAULT_DATA_DIR = Path("data/tool_trajectory_monitoring/distillation_scaling")
DEFAULT_RESULT_DIR = Path("results/tool_trajectory_distillation_mixed_qwen4b")


def validate_mixed_4b_jobs(jobs: list[dict[str, Any]]) -> None:
    """Fail closed unless this is the frozen single 4B transfer job."""
    if len(jobs) != 1:
        raise ValueError("expected exactly one mixed Qwen3.5-4B job")
    job = jobs[0]
    expected = {
        "job_name": JOB_NAME,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "target": "kimi_soft",
        "train_rows": MIXED_ROWS,
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
        "max_steps": -1,
        "num_train_epochs": 1.0,
        "selection_manifest": None,
    }
    for key, value in expected.items():
        if job.get(key) != value:
            raise ValueError(
                f"mixed Qwen3.5-4B job contract drifted for {key}: "
                f"{job.get(key)!r} != {value!r}"
            )
    if job.get("student_rows_sha256") != MIXED_STUDENT_SHA256:
        raise ValueError("mixed student checksum contract drifted")
    if job.get("soft_targets_sha256") != MIXED_SOFT_SHA256:
        raise ValueError("mixed soft-target checksum contract drifted")


def verify_aligned_soft_rows(student_path: Path, soft_path: Path) -> int:
    """Verify exact row alignment without retaining the large prompts in memory."""
    count = 0
    with student_path.open() as students, soft_path.open() as targets:
        while True:
            student_line = students.readline()
            target_line = targets.readline()
            if not student_line and not target_line:
                break
            if not student_line or not target_line:
                raise ValueError("mixed student and soft-target row counts differ")
            student = json.loads(student_line)
            target = json.loads(target_line)
            student_identity = (
                str(student.get("dataset")),
                str(student.get("index")),
                int(student.get("label")),
            )
            target_identity = (
                str(target.get("dataset")),
                str(target.get("index")),
                int(target.get("label")),
            )
            probability = float(target.get("soft_target"))
            if student_identity != target_identity:
                raise ValueError(f"mixed row alignment drift at row {count}")
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"invalid soft target at row {count}")
            count += 1
    if count != MIXED_ROWS:
        raise ValueError(f"expected {MIXED_ROWS} aligned rows, got {count}")
    return count


def make_job(data_dir: Path, result_dir: Path) -> dict[str, Any]:
    student_path = data_dir / "mixed_student_rows.jsonl"
    soft_path = data_dir / "mixed_soft_targets.jsonl"
    output_dir = result_dir / "runs" / JOB_NAME
    job = {
        "job_name": JOB_NAME,
        "design_role": "mixed_data_backbone_size_transfer",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
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
        "train_rows": MIXED_ROWS,
        "max_steps": -1,
        "num_train_epochs": 1.0,
        "student_rows": student_path.as_posix(),
        "student_rows_sha256": MIXED_STUDENT_SHA256,
        "soft_targets": soft_path.as_posix(),
        "soft_targets_sha256": MIXED_SOFT_SHA256,
        "selection_manifest": None,
        "save_steps": math.ceil(MIXED_ROWS / EFFECTIVE_BATCH_SIZE),
        "output_dir": output_dir.as_posix(),
        "causal_adapter_dir": (output_dir / "causal_adapter").as_posix(),
        "model_dir": (output_dir / "model").as_posix(),
    }
    validate_mixed_4b_jobs([job])
    return job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    student_path = args.data_dir / "mixed_student_rows.jsonl"
    soft_path = args.data_dir / "mixed_soft_targets.jsonl"
    if sha256_file(student_path) != MIXED_STUDENT_SHA256:
        raise ValueError("mixed student artifact checksum drifted")
    if sha256_file(soft_path) != MIXED_SOFT_SHA256:
        raise ValueError("mixed soft-target artifact checksum drifted")
    row_count = verify_aligned_soft_rows(student_path, soft_path)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    job = make_job(args.data_dir, args.result_dir)
    jobs_path = args.result_dir / "jobs.jsonl"
    atomic_write_jsonl(jobs_path, [job])
    atomic_write_json(
        args.result_dir / "manifest.json",
        {
            "campaign_id": "tool-trajectory-kimi-soft-mixed-qwen35-4b-v1",
            "hypothesis": (
                "The mixed tool-trajectory plus prior-deception Kimi-soft "
                "intervention transfers to the smaller Qwen3.5-4B backbone."
            ),
            "intervention": (
                "Train Qwen3.5-4B once on all 21,837 mixed rows using the "
                "unchanged rank-128 Qwen3.5-9B soft-distillation recipe."
            ),
            "baseline": (
                "The matched Qwen3.5-9B mixed-data adapter and frozen "
                "unadapted Qwen3.5-4B evaluations; no 4B hyperparameter search."
            ),
            "held_out_selection_rule": (
                "Use the already frozen held-out benchmark suites without "
                "selecting examples, thresholds, or checkpoints on test labels."
            ),
            "arms": "Kimi-soft only; no hard-label training",
            "seed": CAMPAIGN_SEED,
            "student_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
            },
            "recipe": {
                "rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "learning_rate": 5e-5,
                "optimizer": "adamw",
                "loss": "binary soft BCE at selected literal 0|1 logits",
                "quantization": "NF4 double-quantized QLoRA with BF16 compute",
                "micro_batch_size": 1,
                "gradient_accumulation_steps": EFFECTIVE_BATCH_SIZE,
                "effective_batch_size": EFFECTIVE_BATCH_SIZE,
                "max_length": MAX_LENGTH,
                "epochs": 1,
            },
            "tokenizer_equivalence": {
                "qwen35_9b_revision": (
                    "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
                ),
                "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
                "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
                "conclusion": (
                    "The pinned 4B and 9B tokenizer artifacts are byte-identical, "
                    "so the existing 29,696-token zero-truncation audit transfers."
                ),
            },
            "artifacts": {
                "student_rows": {
                    "path": student_path.as_posix(),
                    "rows": row_count,
                    "sha256": MIXED_STUDENT_SHA256,
                },
                "soft_targets": {
                    "path": soft_path.as_posix(),
                    "rows": row_count,
                    "sha256": MIXED_SOFT_SHA256,
                },
                "jobs": {"path": jobs_path.as_posix(), "count": 1},
            },
            "stop_conditions": [
                "either mixed input checksum or row alignment drifts",
                "the target ceases to be Kimi-soft-only BCE",
                "the pinned model or tokenizer revision drifts",
                "FLA 0.5.2 is unavailable or a gated-delta kernel falls back",
                "the rank-128 longest-sequence preflight OOMs",
                "effective batch 32 or the 29,696-token cap changes",
                "the final FP32 causal master or serving rebase is missing",
            ],
        },
    )
    print(f"prepared {JOB_NAME} with {row_count} aligned Kimi-soft rows")


if __name__ == "__main__":
    main()
