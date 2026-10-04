"""Shared execution settings and audits for matched packed LoRA trajectories."""

from __future__ import annotations

import math
import os
import statistics
from pathlib import Path

from gleipnir.flashqla_training import flashqla_environment
from gleipnir.monitoring_systems_screen import gpu_environment
from gleipnir.qwen35_fast_training import (
    DEFAULT_CAUSAL_CONV1D_TARGET,
    DEFAULT_FLA_TARGET,
    DEFAULT_TRITON_TARGET,
    causal_conv1d_environment,
    fla_environment,
    triton_environment,
)


def benchmark_environment(config: dict, root: Path) -> dict[str, str]:
    """Use the same pinned overlays and existing persistent caches for every stage."""
    environment = flashqla_environment(
        triton_environment(
            DEFAULT_TRITON_TARGET,
            causal_conv1d_environment(
                DEFAULT_CAUSAL_CONV1D_TARGET,
                fla_environment(DEFAULT_FLA_TARGET, dict(os.environ)),
            ),
        )
    )
    for overlay in config.get("kernel_overlays", []):
        environment["PYTHONPATH"] += ":" + str(root / overlay)
    environment = gpu_environment(environment, 0, root / config["compiler_cache"])
    if "fa4_cache" in config:
        environment["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"
        environment["FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"] = str(
            root / config["fa4_cache"]
        )
    environment["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
    environment["OMP_NUM_THREADS"] = "4"
    return environment


def summarize(
    metadata: dict,
    warmup: int,
    *,
    accept_learning: bool = False,
    accept_timing: bool = False,
) -> dict:
    """Count every measured update and preserve its physical execution contract."""
    durations = metadata["optimizer_step_timing"]["durations_seconds"]
    if len(durations) != 20 or metadata["training_state"]["global_step"] != 20:
        raise ValueError("incomplete benchmark trajectory")
    if any(not math.isfinite(t) or t <= 0 for t in durations):
        raise ValueError("nonfinite/nonpositive benchmark duration")
    packing = metadata["sequence_packing"]
    if not packing.get("startup_validation"):
        if not packing["preflight"]["passed"] or not all(
            packing[key]["passed"]
            or (
                accept_learning
                and packing[key].get("accepted_for_learning_comparison", False)
            )
            or (
                accept_timing
                and packing.get("attention_backend") == "nvidia_mxfp8"
                and packing.get("timing_authority")
                and packing[key].get("accepted_for_timing_comparison", False)
            )
            for key in ["eager_canary", "compiled_canary"]
        ):
            raise ValueError(
                "fresh packing gates were not passed or explicitly accepted"
            )
    if (
        packing["initial_master_sha256"] == packing["final_master_sha256"]
        or metadata["quantization"]["enabled"]
        or metadata["checkpointed_layer_indices"]
    ):
        raise ValueError("benchmark precision/update/checkpoint contract drift")
    records = metadata["adaptive_microbatching"]["records"]
    if any(r["tokens"] != r["padded_tokens"] for r in records):
        raise ValueError("benchmark introduced padding")
    steady = durations[warmup:]
    return {
        "durations_seconds": durations,
        "measured_total_seconds": sum(steady),
        "measured_mean_seconds": statistics.mean(steady),
        "all_update_seconds": sum(durations),
        "trainer_loop_seconds": metadata["train_metrics"]["train_runtime"],
        "peak_allocated_gib": metadata["peak_cuda_memory_allocated_bytes"] / 2**30,
        "physical_contract": [
            {
                k: row[k]
                for k in ["update", "logical_indices", "tokens", "padded_tokens"]
            }
            for row in records
        ],
        "attention": {
            k: packing.get(k)
            for k in [
                "full_attention",
                "attention_backend",
                "attention_version",
            ]
        },
        "startup_validation": metadata.get("startup_validation"),
        "initial_master_sha256": packing["initial_master_sha256"],
        "compiled": metadata["selective_torch_compile"],
        "packing_gates": {
            key: packing.get(key)
            for key in ["eager_canary", "compiled_canary", "preflight"]
        },
    }
