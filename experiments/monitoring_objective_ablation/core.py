"""Frozen contracts for the monitoring objective ablation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from experiments.monitoring_lr_sweep.core import (
    EXPECTED_CHECKPOINTED_LAYERS,
    EXPECTED_COMPILED_LAYERS,
    MAX_UNIQUE_GRAPHS,
    MODEL_ID,
    MODEL_REVISION,
    SOFT_TARGETS_SHA256,
)

CAMPAIGN_ID = "gleipnir4b-monitoring-objective-ablation-v1"
TRAIN_ROWS = 8_688
EFFECTIVE_BATCH_SIZE = 32
OPTIMIZER_STEPS = math.ceil(TRAIN_ROWS / EFFECTIVE_BATCH_SIZE)
LEARNING_RATE = 2e-5
RANK = 128
LORA_ALPHA = 256
MAX_LENGTH = 29_696
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
RAW_SOURCE_SHA256 = "43bfc9f78d6584c43135109636f5e1747c694842419a327c3f37614fb9301802"

ARM_SPECS: tuple[dict[str, Any], ...] = (
    {
        "job_name": "soft-rationale-w005",
        "kind": "rationale",
        "completion_loss_weight": 0.05,
        "sequential_objective_backward": True,
        "mil_loss_weight": 0.0,
        "mil_pooling": "logmeanexp",
        "gradient_checkpointing_policy": "all",
        "selective_torch_compile_policy": (
            "checkpointed_full_attention_and_linear_shell"
        ),
    },
    {
        "job_name": "soft-rationale-w020",
        "kind": "rationale",
        "completion_loss_weight": 0.20,
        "sequential_objective_backward": True,
        "mil_loss_weight": 0.0,
        "mil_pooling": "logmeanexp",
        "gradient_checkpointing_policy": "all",
        "selective_torch_compile_policy": (
            "checkpointed_full_attention_and_linear_shell"
        ),
    },
    {
        "job_name": "soft-mil-max-w025",
        "kind": "mil",
        "completion_loss_weight": 0.0,
        "sequential_objective_backward": False,
        "mil_loss_weight": 0.25,
        "mil_pooling": "max",
        "gradient_checkpointing_policy": "linear_attention_only",
        "selective_torch_compile_policy": "full_attention_and_linear_shell",
    },
    {
        "job_name": "soft-mil-lme-w025",
        "kind": "mil",
        "completion_loss_weight": 0.0,
        "sequential_objective_backward": False,
        "mil_loss_weight": 0.25,
        "mil_pooling": "logmeanexp",
        "gradient_checkpointing_policy": "linear_attention_only",
        "selective_torch_compile_policy": "full_attention_and_linear_shell",
    },
    {
        "job_name": "soft-mil-top3-w025",
        "kind": "mil",
        "completion_loss_weight": 0.0,
        "sequential_objective_backward": False,
        "mil_loss_weight": 0.25,
        "mil_pooling": "topk_mean",
        "gradient_checkpointing_policy": "linear_attention_only",
        "selective_torch_compile_policy": "full_attention_and_linear_shell",
    },
)

CONTROL = {
    "job_name": "adamw-lr2em05-seed0",
    "macro_pauroc_at_20": 0.8710662298082067,
    "macro_auroc": 0.9576432954716307,
    "macro_brier": 0.07927280281324776,
    "source_pauroc_at_20": {
        "gloom_exfiltration": 0.7838325906558521,
        "test_stride": 0.9582998689605613,
    },
}


def expected_job_names() -> list[str]:
    return [str(spec["job_name"]) for spec in ARM_SPECS]


def validate_jobs(jobs: list[dict[str, Any]]) -> None:
    """Reject drift from the predeclared five-arm screen."""
    if [str(job.get("job_name")) for job in jobs] != expected_job_names():
        raise ValueError("objective-ablation job order drifted")
    specs = {str(spec["job_name"]): spec for spec in ARM_SPECS}
    for job in jobs:
        spec = specs[str(job["job_name"])]
        expected = {
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "seed": 0,
            "rank": RANK,
            "lora_alpha": LORA_ALPHA,
            "max_length": MAX_LENGTH,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_train_epochs": 1.0,
            "max_steps": -1,
            "soft_loss_weight": 1.0,
            "direct_loss_weight": 0.0,
            "completion_logits_mode": "selected_positions",
            "completion_projection_chunk_size": 128,
            "sequential_objective_backward": spec["sequential_objective_backward"],
            "completion_loss_weight": spec["completion_loss_weight"],
            "mil_loss_weight": spec["mil_loss_weight"],
            "mil_pooling": spec["mil_pooling"],
            "mil_temperature": 1.0,
            "mil_top_k": 3,
            "mil_max_instances": 8,
            "gradient_checkpointing_policy": spec["gradient_checkpointing_policy"],
            "selective_torch_compile_policy": spec["selective_torch_compile_policy"],
            "train_rows": TRAIN_ROWS,
            "soft_targets_sha256": SOFT_TARGETS_SHA256,
        }
        for key, value in expected.items():
            if job.get(key) != value:
                raise ValueError(
                    f"job contract drift for {job.get('job_name')!r} {key}: "
                    f"{job.get(key)!r} != {value!r}"
                )
        if len(str(job.get("student_rows_sha256", ""))) != 64:
            raise ValueError("rationale-enriched student checksum is missing")
    if len({str(job["student_rows_sha256"]) for job in jobs}) != 1:
        raise ValueError("objective arms do not share one enriched student artifact")


def validate_training_metadata(path: Path, job: dict[str, Any]) -> dict[str, Any]:
    """Require objective identity and the proven fast-kernel training recipe."""
    import json

    metadata = json.loads(path.read_text())
    losses = metadata.get("losses", {})
    expected_losses = {
        "completion_weight": float(job["completion_loss_weight"]),
        "completion_logits_mode": "selected_positions",
        "completion_projection_chunk_size": 128,
        "sequential_objective_backward": bool(job["sequential_objective_backward"]),
        "direct_weight": 0.0,
        "pairwise_weight": 0.0,
        "soft_weight": 1.0,
        "soft_type": "bce",
        "mil_weight": float(job["mil_loss_weight"]),
        "mil_pooling": str(job["mil_pooling"]),
        "mil_temperature": 1.0,
        "mil_top_k": 3,
        "mil_max_instances": 8,
    }
    if losses != expected_losses:
        raise ValueError(f"loss metadata drifted: {losses} != {expected_losses}")
    fla = metadata.get("flash_linear_attention", {})
    causal = metadata.get("causal_conv1d", {})
    compiled = metadata.get("selective_torch_compile", {})
    batch = metadata.get("training_batch", {})
    if (
        fla.get("available") is not True
        or fla.get("required") is not True
        or fla.get("version") != "0.5.2"
        or fla.get("backend_dispatch_disabled") is not True
        or fla.get("triton_version") != "3.7.1"
    ):
        raise ValueError("pinned FLA metadata is missing")
    if (
        causal.get("available") is not True
        or causal.get("required") is not True
        or causal.get("version") != "1.6.2.post1"
    ):
        raise ValueError("pinned causal-conv1d metadata is missing")
    if batch != {
        "effective_batch_size": 32,
        "gradient_accumulation_steps": 32,
        "micro_batch_size": 1,
    }:
        raise ValueError("training batch drifted")
    expected_checkpointed = (
        list(range(32)) if job["kind"] == "rationale" else EXPECTED_CHECKPOINTED_LAYERS
    )
    if metadata.get("checkpointed_layer_indices") != expected_checkpointed:
        raise ValueError("checkpointing policy drifted")
    if (
        metadata.get("gradient_checkpointing_policy")
        != job["gradient_checkpointing_policy"]
    ):
        raise ValueError("checkpointing policy identity drifted")
    if compiled.get("policy") != job["selective_torch_compile_policy"]:
        raise ValueError("selective compilation policy identity drifted")
    if compiled.get("compiled_layer_indices") != EXPECTED_COMPILED_LAYERS:
        raise ValueError("compiled layer set drifted")
    unique_graphs = int(
        compiled.get("dynamo_counters", {}).get("stats", {}).get("unique_graphs", 0)
    )
    if not 1 <= unique_graphs <= MAX_UNIQUE_GRAPHS:
        raise ValueError(f"unexpected Dynamo graph count: {unique_graphs}")
    if (
        int(metadata.get("training_state", {}).get("global_step", -1))
        != OPTIMIZER_STEPS
    ):
        raise ValueError("optimizer-step count drifted")
    expected_inputs = {
        "completion": bool(job["completion_loss_weight"]),
        "direct": True,
    }
    if metadata.get("materialized_training_inputs") != expected_inputs:
        raise ValueError("training input materialization drifted")
    return metadata


def select_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the same practical-effect gate used for the learning-rate screen."""
    eligible = []
    for row in rows:
        source_deltas = {
            source: float(score) - float(CONTROL["source_pauroc_at_20"][source])
            for source, score in row["source_pauroc_at_20"].items()
        }
        deltas = {
            "macro_pauroc_at_20": float(row["macro_pauroc_at_20"])
            - float(CONTROL["macro_pauroc_at_20"]),
            "macro_auroc": float(row["macro_auroc"]) - float(CONTROL["macro_auroc"]),
            "macro_brier": float(row["macro_brier"]) - float(CONTROL["macro_brier"]),
            "source_pauroc_at_20": source_deltas,
        }
        row["control_deltas"] = deltas
        row["eligible"] = bool(
            deltas["macro_pauroc_at_20"] >= 0.005
            and min(source_deltas.values()) >= -0.01
            and deltas["macro_brier"] <= 0.005
        )
        if row["eligible"]:
            eligible.append(row)
    selected = max(
        eligible,
        key=lambda row: (
            float(row["macro_pauroc_at_20"]),
            float(row["macro_auroc"]),
            -float(row["macro_brier"]),
        ),
        default=None,
    )
    return {
        "selected_job": None if selected is None else str(selected["job_name"]),
        "control_retained": selected is None,
        "eligible_jobs": [str(row["job_name"]) for row in eligible],
    }
