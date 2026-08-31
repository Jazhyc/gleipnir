#!/usr/bin/env python3
"""Freeze the one-job Qwen3.5-0.8B mixed-data soft-distillation follow-up."""

from __future__ import annotations

import argparse
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
from experiments.tool_trajectory_monitoring.prepare_mixed_4b_distillation import (
    MIXED_ROWS,
    MIXED_SOFT_SHA256,
    MIXED_STUDENT_SHA256,
    TOKENIZER_JSON_SHA256,
    verify_aligned_soft_rows,
)
from gleipnir.qwen35_adapter_rebase import sha256_file

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_CONFIG_SHA256 = "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"
TOKENIZER_CONFIG_SHA256 = (
    "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c"
)
CHAT_TEMPLATE_SHA256 = (
    "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
)
JOB_NAME = "soft-n21837-mixed-qwen35-08b-seed0"
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE
DEFAULT_DATA_DIR = Path("data/tool_trajectory_monitoring/distillation_scaling")
DEFAULT_RESULT_DIR = Path("results/tool_trajectory_distillation_mixed_qwen08b")


def validate_mixed_08b_jobs(jobs: list[dict[str, Any]]) -> None:
    """Fail closed unless this is the frozen single 0.8B transfer job."""
    if len(jobs) != 1:
        raise ValueError("expected exactly one mixed Qwen3.5-0.8B job")
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
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
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
                f"mixed Qwen3.5-0.8B job contract drifted for {key}: "
                f"{job.get(key)!r} != {value!r}"
            )
    if job.get("student_rows_sha256") != MIXED_STUDENT_SHA256:
        raise ValueError("mixed student checksum contract drifted")
    if job.get("soft_targets_sha256") != MIXED_SOFT_SHA256:
        raise ValueError("mixed soft-target checksum contract drifted")


def make_job(data_dir: Path, result_dir: Path) -> dict[str, Any]:
    student_path = data_dir / "mixed_student_rows.jsonl"
    soft_path = data_dir / "mixed_soft_targets.jsonl"
    output_dir = result_dir / "runs" / JOB_NAME
    job = {
        "job_name": JOB_NAME,
        "design_role": "mixed_data_lower_bound_backbone_transfer",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": CAMPAIGN_SEED,
        "target": "kimi_soft",
        "rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "max_length": MAX_LENGTH,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
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
    validate_mixed_08b_jobs([job])
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
            "campaign_id": "tool-trajectory-kimi-soft-mixed-qwen35-08b-v1",
            "status": "frozen_before_training",
            "hypothesis": (
                "The complete mixed Kimi-soft intervention provides a larger "
                "OOD uplift at Qwen3.5-0.8B than at larger Qwen3.5 backbones, "
                "pushing the low-cost monitoring frontier downward."
            ),
            "intervention": (
                "Train Qwen3.5-0.8B once on all 21,837 mixed rows with the "
                "rank-128 soft-only recipe and effective batch 32."
            ),
            "baselines": [
                "the frozen unadapted Qwen3.5-0.8B under the student interface",
                "the completed mixed Qwen3.5-4B and Qwen3.5-9B adapters",
            ],
            "held_out_selection_rule": (
                "Evaluate the final one-epoch checkpoint and unadapted base on "
                "all 6,395 frozen OOD rows without selecting prompts, examples, "
                "thresholds, or checkpoints on OOD labels."
            ),
            "arms": "Kimi-soft only; no hard-label or completion-loss training",
            "seed": CAMPAIGN_SEED,
            "student_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "config_sha256": MODEL_CONFIG_SHA256,
            },
            "recipe": {
                "rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "learning_rate": 5e-5,
                "optimizer": "adamw",
                "loss": "binary soft BCE at selected literal 0|1 logits",
                "quantization": "NF4 double-quantized QLoRA with BF16 compute",
                "micro_batch_size": MICRO_BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "effective_batch_size": EFFECTIVE_BATCH_SIZE,
                "max_length": MAX_LENGTH,
                "epochs": 1,
            },
            "tokenizer_compatibility": {
                "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
                "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
                "chat_template_sha256": CHAT_TEMPLATE_SHA256,
                "conclusion": (
                    "The tokenizer vocabulary is byte-identical to 4B/9B. The "
                    "0.8B template defaults to non-thinking and renders the same "
                    "explicit enable_thinking=false assistant boundary."
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
                "the objective ceases to be Kimi-soft-only BCE",
                "the pinned model revision or tokenizer vocabulary drifts",
                "FLA 0.5.2 is unavailable or a gated-delta kernel falls back",
                "the rank-128 longest-sequence microbatch-1 preflight OOMs",
                "effective batch 32 or the 29,696-token cap changes",
                "the final FP32 causal master or serving rebase is missing",
                "base or adapter OOD coverage/parity fails",
            ],
        },
    )
    print(f"prepared {JOB_NAME} with {row_count} aligned Kimi-soft rows")


if __name__ == "__main__":
    main()
