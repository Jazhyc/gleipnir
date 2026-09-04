"""Frozen contracts for the monitoring-only Gleipnir 4B LR screen."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from experiments.optimizer_lr_scaling.core import learning_rate_tag

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LEARNING_RATES = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4)
CONTROL_LEARNING_RATE = 5e-5
TRAINING_SEED = 0
TRAIN_ROWS = 8_688
STUDENT_ROWS_SHA256 = "03669ac12452a2b63fe13193611fb99bfaeeb2456e6aaeb0c1be43471ea4387c"
SOFT_TARGETS_SHA256 = "1ae8c3cccc2546335f8002d1475cd86d7a7e059fedb66345d1aa13d6a30a526a"
RANK = 128
LORA_ALPHA = 256
MAX_LENGTH = 29_696
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
EFFECTIVE_BATCH_SIZE = 32
OPTIMIZER_STEPS = math.ceil(TRAIN_ROWS / EFFECTIVE_BATCH_SIZE)
GRADIENT_CHECKPOINTING_POLICY = "linear_attention_only"
SELECTIVE_TORCH_COMPILE_POLICY = "full_attention_and_linear_shell"
EXPECTED_CHECKPOINTED_LAYERS = [
    index for index in range(32) if (index + 1) % 4 != 0
]
EXPECTED_COMPILED_LAYERS = list(range(32))
MAX_UNIQUE_GRAPHS = 16


def job_name(learning_rate: float) -> str:
    """Return the stable seed-0 AdamW job name for one learning rate."""
    return f"adamw-lr{learning_rate_tag(learning_rate)}-seed{TRAINING_SEED}"


def expected_job_names() -> list[str]:
    return [job_name(rate) for rate in LEARNING_RATES]


def validate_jobs(jobs: list[dict[str, Any]]) -> None:
    """Reject any drift from the five-cell monitoring-only design."""
    if [str(job.get("job_name")) for job in jobs] != expected_job_names():
        raise ValueError("learning-rate jobs differ from the frozen ordered grid")
    if [float(job.get("learning_rate", -1)) for job in jobs] != list(LEARNING_RATES):
        raise ValueError("learning-rate grid drifted")
    for job in jobs:
        expected = {
            "capacity_kind": "lora",
            "data_scope": "monitoring_only",
            "deception_rows": 0,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "seed": TRAINING_SEED,
            "target": "kimi_soft",
            "rank": RANK,
            "lora_alpha": LORA_ALPHA,
            "max_length": MAX_LENGTH,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "gradient_checkpointing": True,
            "gradient_checkpointing_policy": GRADIENT_CHECKPOINTING_POLICY,
            "selective_torch_compile_policy": SELECTIVE_TORCH_COMPILE_POLICY,
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
            "require_causal_conv1d": True,
            "fla_disable_backend_dispatch": True,
            "triton_version": "3.7.1",
            "train_rows": TRAIN_ROWS,
            "selection_manifest": None,
            "student_rows_sha256": STUDENT_ROWS_SHA256,
            "soft_targets_sha256": SOFT_TARGETS_SHA256,
        }
        for key, value in expected.items():
            if job.get(key) != value:
                raise ValueError(
                    f"job contract drift for {job.get('job_name')!r} {key}: "
                    f"{job.get(key)!r} != {value!r}"
                )


def learning_rate_lanes(jobs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Schedule the control and nearest candidate first across two equal lanes."""
    validate_jobs(jobs)
    priority = (5e-5, 1e-4, 2e-5, 2e-4, 1e-5)
    by_rate = {float(job["learning_rate"]): job for job in jobs}
    lanes: list[list[dict[str, Any]]] = [[], []]
    for index, rate in enumerate(priority):
        lanes[index % 2].append(by_rate[rate])
    return lanes


def validate_training_metadata(
    path: Path,
    learning_rate: float,
    *,
    expected_steps: int = OPTIMIZER_STEPS,
    require_canary: bool = False,
) -> dict[str, Any]:
    """Require the proven Qwen3.5 fast-kernel QLoRA recipe after every run."""
    metadata = json.loads(path.read_text())
    fla = metadata.get("flash_linear_attention", {})
    causal = metadata.get("causal_conv1d", {})
    quantization = metadata.get("quantization", {})
    batch = metadata.get("training_batch", {})
    optimization = metadata.get("optimization", {})
    compiled = metadata.get("selective_torch_compile", {})
    if (
        fla.get("available") is not True
        or fla.get("required") is not True
        or fla.get("version") != "0.5.2"
        or fla.get("backend_dispatch_disabled") is not True
        or fla.get("triton_version") != "3.7.1"
        or not metadata.get("gated_delta_kernel_modules")
    ):
        raise ValueError("training metadata does not prove pinned FLA use")
    if (
        causal.get("available") is not True
        or causal.get("required") is not True
        or causal.get("version") != "1.6.2.post1"
        or not causal.get("kernel_modules")
        or any(
            not str(module).startswith("causal_conv1d.")
            for module in causal.get("kernel_modules", [])
        )
    ):
        raise ValueError("training metadata does not prove causal-conv1d use")
    if quantization != {
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "enabled": True,
    }:
        raise ValueError("QLoRA quantization metadata drifted")
    if batch != {
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "micro_batch_size": MICRO_BATCH_SIZE,
    }:
        raise ValueError("training batch metadata drifted")
    if (
        metadata.get("gradient_checkpointing") is not True
        or metadata.get("gradient_checkpointing_policy")
        != GRADIENT_CHECKPOINTING_POLICY
        or metadata.get("checkpointed_layer_indices")
        != EXPECTED_CHECKPOINTED_LAYERS
    ):
        raise ValueError("selective checkpointing metadata drifted")
    if (
        compiled.get("policy") != SELECTIVE_TORCH_COMPILE_POLICY
        or compiled.get("compiled_layer_indices") != EXPECTED_COMPILED_LAYERS
        or compiled.get("disabled_linear_attention_layer_indices")
        != EXPECTED_CHECKPOINTED_LAYERS
        or compiled.get("backend") != "inductor"
        or compiled.get("mode") != "default"
        or compiled.get("dynamic") is not True
        or compiled.get("global_torch_compile") is not False
    ):
        raise ValueError("selective compilation metadata drifted")
    unique_graphs = int(
        compiled.get("dynamo_counters", {}).get("stats", {}).get("unique_graphs", 0)
    )
    if not 1 <= unique_graphs <= MAX_UNIQUE_GRAPHS:
        raise ValueError(f"unexpected Dynamo graph count: {unique_graphs}")
    if require_canary and compiled.get("canary", {}).get("passed") is not True:
        raise ValueError("same-weights compile canary did not pass")
    if not math.isclose(float(optimization.get("learning_rate", -1)), learning_rate):
        raise ValueError("recorded learning rate drifted")
    if (
        optimization.get("optimizer") != "adamw"
        or optimization.get("trainer_optim") != "adamw_torch"
    ):
        raise ValueError("recorded optimizer drifted")
    if metadata.get("direct_logits_mode") != "selected_positions":
        raise ValueError("selected-position logits contract drifted")
    if metadata.get("materialized_training_inputs") != {
        "completion": False,
        "direct": True,
    }:
        raise ValueError("direct-only input materialization drifted")
    completed = int(metadata.get("training_state", {}).get("global_step", -1))
    if completed != expected_steps:
        raise ValueError(f"completed {completed} steps instead of {expected_steps}")
    return metadata


def select_screen_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen practical-effect and regression guardrails."""
    if {float(row["learning_rate"]) for row in rows} != set(LEARNING_RATES):
        raise ValueError("summary rows do not cover the frozen learning-rate grid")
    control = next(
        row for row in rows if float(row["learning_rate"]) == CONTROL_LEARNING_RATE
    )
    control_sources = dict(control["source_pauroc_at_20"])
    eligible = []
    for row in rows:
        source_deltas = {
            source: float(score) - float(control_sources[source])
            for source, score in row["source_pauroc_at_20"].items()
        }
        primary_gain = float(row["macro_pauroc_at_20"]) - float(
            control["macro_pauroc_at_20"]
        )
        brier_regression = float(row["macro_brier"]) - float(control["macro_brier"])
        row["control_deltas"] = {
            "macro_pauroc_at_20": primary_gain,
            "macro_auroc": float(row["macro_auroc"]) - float(control["macro_auroc"]),
            "macro_brier": brier_regression,
            "source_pauroc_at_20": source_deltas,
        }
        row["eligible_to_replace_control"] = bool(
            primary_gain >= 0.005
            and min(source_deltas.values()) >= -0.01
            and brier_regression <= 0.005
        )
        if row["eligible_to_replace_control"]:
            eligible.append(row)
    selected = max(
        eligible,
        key=lambda row: (
            float(row["macro_pauroc_at_20"]),
            float(row["macro_auroc"]),
            -float(row["macro_brier"]),
        ),
        default=control,
    )
    return {
        "selected_learning_rate": float(selected["learning_rate"]),
        "selected_job": str(selected["job_name"]),
        "control_retained": selected is control,
        "eligible_jobs": [str(row["job_name"]) for row in eligible],
        "rule": {
            "minimum_macro_pauroc_at_20_gain": 0.005,
            "maximum_per_source_pauroc_at_20_regression": 0.01,
            "maximum_macro_brier_regression": 0.005,
            "ranking": "macro pAUROC@20, then macro AUROC, then lower macro Brier",
        },
    }
