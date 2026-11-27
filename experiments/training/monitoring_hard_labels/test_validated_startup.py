"""Validate truthful reuse receipts and kernel installation without model probes."""

import json
from types import SimpleNamespace

import pytest

from gleipnir import validated_startup


def reference_metadata():
    return {
        "model": "Qwen/Qwen3.5-4B",
        "model_revision": "revision",
        "training_state": {"global_step": 272},
        "quantization": {"enabled": False},
        "gradient_checkpointing": False,
        "gated_delta_backend": {
            "backend": "flashqla",
            "replaced_layers": 24,
            "boundary_policy": "bf16_fp32_gates_norm",
            "auto_cp": False,
            "finite": True,
            "passed": False,
            "revision": "pinned",
        },
        "sequence_packing": {
            "max_packed_tokens": 16384,
            **{
                k: {"passed": True}
                for k in ("eager_canary", "compiled_canary", "preflight")
            },
        },
    }


def test_reuse_keeps_previous_failure_and_does_not_claim_a_fresh_pass(tmp_path):
    path = tmp_path / "training_metadata.json"
    path.write_text(json.dumps(reference_metadata()))
    reference = validated_startup.validation_reference(path)
    assert reference["performed_this_run"] is False
    assert reference["reference_backend"]["passed"] is False
    reused = validated_startup.skipped_diagnostic(reference, "preflight")
    assert "passed" not in reused
    assert reused["reference_result"]["passed"] is True
    metadata = reference_metadata()
    metadata["sequence_packing"]["eager_canary"]["passed"] = False
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="does not validate"):
        validated_startup.validation_reference(path)


def test_install_only_binds_kernel_without_running_model(monkeypatch):
    modules = [SimpleNamespace(chunk_gated_delta_rule=None) for _ in range(24)]
    model = SimpleNamespace(modules=lambda: modules)
    reference = {
        "reference_sha256": "receipt",
        "reference_backend": {"revision": "pinned"},
    }
    monkeypatch.setattr(
        validated_startup,
        "load_flashqla",
        lambda: (lambda: None, {"revision": "pinned"}),
    )
    receipt = validated_startup.install_validated_flashqla(model, reference)
    assert all(callable(module.chunk_gated_delta_rule) for module in modules)
    assert receipt["performed_this_run"] is False
    assert "passed" not in receipt and "finite" not in receipt
    assert receipt["replaced_layers"] == 24
