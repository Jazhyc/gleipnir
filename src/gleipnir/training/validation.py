"""Validate completed training against an explicitly resolved profile."""

from pathlib import Path


def validate_training_metadata(
    metadata: dict, job: dict, initial_hash: str, *, expected_steps: int, profile: dict
) -> None:
    """Require completed updates, exact loss weights and the selected packed recipe."""
    from gleipnir.campaigns.systems_screen import nested_value

    reused = metadata.get("startup_validation")
    skipped_paths = {
        "sequence_packing.eager_canary.passed",
        "sequence_packing.compiled_canary.passed",
        "sequence_packing.preflight.passed",
        "gated_delta_backend.finite",
    }
    if reused:
        from gleipnir.training.startup import (
            reused_diagnostic_view,
            validation_reference,
        )

        expected_reference = validation_reference(
            Path(job["startup_validation_reference"]),
            packed_attention_backend=job.get("packed_attention_backend", "sdpa"),
            packed_attention_version=job.get("packed_attention_version"),
            learning_gradient_tolerance=job.get("packing_learning_gradient_tolerance"),
            expected_sha256=job.get("startup_validation_reference_sha256"),
            **({"native_fp4_mlp": True} if job.get("native_fp4_mlp", False) else {}),
        )
        if reused["reference_sha256"] != expected_reference["reference_sha256"]:
            raise ValueError("reused validation identity drift")
        metadata = reused_diagnostic_view(
            {**metadata, "startup_validation": expected_reference}
        )
    for path, expected in profile["metadata_expectations"].items():
        if reused and path in skipped_paths:
            continue
        if nested_value(metadata, path) != expected:
            raise ValueError(f"systems metadata drift: {path}")
    packing = metadata["sequence_packing"]
    if metadata["training_state"]["global_step"] != expected_steps or (
        not reused
        and not all(
            packing[k]["passed"]
            for k in ("eager_canary", "compiled_canary", "preflight")
        )
    ):
        raise ValueError("completed training or packing gates failed")
    if packing["initial_master_sha256"] != initial_hash:
        raise ValueError("initial adapter identity drift")
    if packing["initial_master_sha256"] == packing["final_master_sha256"]:
        raise ValueError("master adapter did not change")
    losses = metadata["losses"]
    if (
        losses["soft_weight"] != job["soft_loss_weight"]
        or losses["direct_weight"] != job["direct_loss_weight"]
        or losses["completion_weight"] != 0
        or losses["accumulation_policy"] != "sum_per_example_over_logical_batch_v1"
    ):
        raise ValueError("objective or equal-example weighting drift")
    if (
        metadata["gradient_checkpointing"]
        or metadata["quantization"]["enabled"]
        or metadata["direct_logits_mode"] != "selected_positions"
        or metadata["optimization"]["learning_rate"] != job["learning_rate"]
        or metadata["gated_delta_backend"]["backend"] != "flashqla"
    ):
        raise ValueError("selected packed BF16 recipe drift")
