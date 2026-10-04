"""Reuse only the accepted BF16 FA4 contract without relabeling strict failure."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.monitoring_hard_labels.test_validated_startup import reference_metadata
from gleipnir import validated_startup
from gleipnir.attention_backends import packed_fa4_environment
from gleipnir.packed_training import validate_packed_training_config
from tests.test_packed_training import student_config


def fa4_reference():
    metadata = reference_metadata()
    metadata["quantization"]["full_bf16_lora"] = {"verified": True}
    packing = metadata["sequence_packing"]
    packing.update(
        attention_backend="flash_attention_4",
        attention_version="4.0.0b33",
        learning_gradient_tolerance=0.10,
    )
    for key, error in [("eager_canary", 0.083), ("compiled_canary", 0.052)]:
        packing[key] = {
            "passed": False,
            "accepted_for_learning_comparison": True,
            "learning_gradient_tolerance": 0.10,
            "adapter_gradient_relative_l2": error,
            "independent_loss": 0.949,
            "packed_loss": 0.958,
            "cases": [
                {
                    "repeat_max_abs": 0.0,
                    "perturb_max_abs": 0.0,
                    "cross_input_grad_max_abs": 0.0,
                    "own_input_grad_max_abs": 1.0,
                }
            ],
        }
    return metadata


def write_reference(tmp_path, metadata):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(metadata))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_accepted_fa4_reuse_preserves_failure_and_skipped_checks(tmp_path):
    metadata = fa4_reference()
    path, digest = write_reference(tmp_path, metadata)
    ref = validated_startup.validation_reference(
        path,
        packed_attention_backend="flash_attention_4",
        packed_attention_version="4.0.0b33",
        learning_gradient_tolerance=0.10,
        expected_sha256=digest,
    )
    assert not ref["performed_this_run"]
    assert not ref["reference_packing"]["eager_canary"]["passed"]
    assert ref["reference_packing"]["eager_canary"]["accepted_for_learning_comparison"]
    raw = {
        **metadata,
        "startup_validation": ref,
        "gated_delta_backend": {"performed_this_run": False},
        "sequence_packing": {
            **metadata["sequence_packing"],
            "eager_canary": validated_startup.skipped_diagnostic(ref, "eager_canary"),
        },
    }
    view = validated_startup.reused_diagnostic_view(raw)
    assert not view["sequence_packing"]["eager_canary"]["passed"]
    assert "passed" not in raw["sequence_packing"]["eager_canary"]
    assert view["gated_delta_backend"]["finite"]
    assert json.loads(path.read_text()) == metadata


@pytest.mark.parametrize(
    "problem", ["leakage", "gradient", "loss", "nf4", "version", "checkpointing"]
)
def test_fa4_reference_rejects_false_acceptance_or_recipe_drift(tmp_path, problem):
    metadata = fa4_reference()
    gate = metadata["sequence_packing"]["eager_canary"]
    if problem == "leakage":
        gate["cases"][0]["cross_input_grad_max_abs"] = 0.001
    elif problem == "gradient":
        gate["adapter_gradient_relative_l2"] = 0.11
    elif problem == "loss":
        gate["packed_loss"] = 2.0
    elif problem == "nf4":
        metadata["quantization"]["enabled"] = True
    elif problem == "version":
        metadata["sequence_packing"]["attention_version"] = "other"
    else:
        metadata["gradient_checkpointing"] = True
    path, _ = write_reference(tmp_path, metadata)
    with pytest.raises(ValueError):
        validated_startup.validation_reference(path)


def test_reuse_rejects_checksum_and_backend_mismatch(tmp_path):
    path, digest = write_reference(tmp_path, reference_metadata())
    with pytest.raises(ValueError, match="checksum"):
        validated_startup.validation_reference(path, expected_sha256="wrong")
    with pytest.raises(ValueError, match="attention/acceptance"):
        validated_startup.validation_reference(
            path,
            packed_attention_backend="flash_attention_4",
            packed_attention_version="4.0.0b33",
            learning_gradient_tolerance=0.10,
            expected_sha256=digest,
        )
    config = student_config()
    config["training"].update(
        packed_attention_backend="flash_attention_4",
        packed_attention_version="4.0.0b33",
        packing_learning_gradient_tolerance=0.10,
        startup_validation_reference=str(path),
        startup_validation_reference_sha256=digest,
    )
    assert validate_packed_training_config(config)


def test_runtime_change_requires_new_validation(tmp_path, monkeypatch):
    import torch

    path, _ = write_reference(tmp_path, fa4_reference())
    monkeypatch.setattr(validated_startup, "package_version", lambda _: "wrong")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "NVIDIA B200")
    with pytest.raises(ValueError, match="hardware/software"):
        validated_startup.validation_reference(path, verify_runtime=True)


def test_fa4_overlay_keeps_flashqla_first_and_reuses_cache():
    original = {"PYTHONPATH": "flashqla:fla:project", "CUDA_VISIBLE_DEVICES": "0"}
    frozen = deepcopy(original)
    first = packed_fa4_environment(original, Path("/workspace/gleipnir"))
    second = packed_fa4_environment(first, Path("/workspace/gleipnir"))
    assert first == second and original == frozen
    assert (
        first["PYTHONPATH"]
        == "flashqla:fla:project:/workspace/gleipnir/.cache/kernels/fa4"
    )
    assert (
        first["FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"]
        == "/workspace/gleipnir/.cache/training/fa4_4.0.0b33_cute"
    )
