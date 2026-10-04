"""Check native layout, causal isolation, restoration and matched launch contracts."""

import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as functional
import yaml

from experiments.b200_bf16_fa4.run import benchmark_job
from gleipnir.packed_sequences import (
    PackedSequenceLayout,
    installed_segmented_sdpa,
    packed_fa4_interface,
    segmented_sdpa_interface,
)
from gleipnir.packed_training import validate_packed_training_config
from tests.test_packed_training import student_config


def test_failed_packing_receipt_survives_the_original_exception(tmp_path):
    import json

    from gleipnir.packed_training import record_packing_canary
    from gleipnir.packed_training_screen import PackingCanaryError

    receipt = {"passed": False, "adapter_gradient_relative_l2": 0.0829}
    error = PackingCanaryError("strict gate failed", receipt)
    metadata = {"enabled": True}

    def check():
        raise error

    output = tmp_path / "nested" / "packing_canary.json"
    with pytest.raises(PackingCanaryError) as caught:
        record_packing_canary(check, metadata, "eager_canary", output)
    assert caught.value is error
    assert json.loads(output.read_text())["eager_canary"] == receipt
    assert metadata["eager_canary"] == receipt


@pytest.mark.parametrize("fail", [False, True])
def test_fa4_router_restores_binding_after_success_or_failure(monkeypatch, fail):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from gleipnir import attention_backends

    module = ModuleType("flash_attn.cute")
    module.flash_attn_varlen_func = varlen
    monkeypatch.setitem(sys.modules, "flash_attn.cute", module)
    monkeypatch.setattr(attention_backends, "attention_loader_kwargs", lambda *args: {})
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    try:
        with installed_segmented_sdpa("flash_attention_4", "4.0.0b33"):
            assert ALL_ATTENTION_FUNCTIONS["sdpa"] is not original
            assert ALL_ATTENTION_FUNCTIONS["sdpa"]._gleipnir_packed_boundaries
            if fail:
                raise RuntimeError("diagnostic failure")
    except RuntimeError:
        assert fail
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original


def sdpa(module, q, k, v, mask, **kwargs):
    return functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=True
    ).transpose(1, 2), None


def varlen(q, k, v, **kwargs):
    assert kwargs["causal"]
    assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_k"] == 5
    cuts = kwargs["cu_seqlens_q"].tolist()
    return torch.cat(
        [
            sdpa(
                None,
                q[a:b].transpose(0, 1).unsqueeze(0),
                k[a:b].transpose(0, 1).unsqueeze(0),
                v[a:b].transpose(0, 1).unsqueeze(0),
                None,
            )[0][0]
            for a, b in zip(cuts[:-1], cuts[1:], strict=True)
        ]
    ), None


def test_varlen_layout_matches_reference_and_preserves_backward_isolation():
    torch.manual_seed(0)
    q, k, v = [torch.randn(1, h, 8, 4, requires_grad=True) for h in [4, 2, 2]]
    kwargs = PackedSequenceLayout((3, 5)).kernel_kwargs()
    actual, _ = packed_fa4_interface(sdpa, varlen)(None, q, k, v, None, **kwargs)
    expected, _ = segmented_sdpa_interface(sdpa)(None, q, k, v, None, **kwargs)
    torch.testing.assert_close(actual, expected)
    actual[:, 3:].square().sum().backward()
    for tensor in [q, k, v]:
        assert torch.count_nonzero(tensor.grad[:, :, :3]) == 0
        assert torch.count_nonzero(tensor.grad[:, :, 3:]) > 0
    altered = q.detach().clone()
    altered[:, :, :3] += 100
    after, _ = packed_fa4_interface(sdpa, varlen)(None, altered, k, v, None, **kwargs)
    torch.testing.assert_close(actual[:, 3:], after[:, 3:], atol=0, rtol=0)


@pytest.mark.parametrize("problem", ["mask", "boundaries", "dtype", "dropout"])
def test_invalid_native_layout_fails_closed(problem):
    kwargs = PackedSequenceLayout((3, 5)).kernel_kwargs()
    q = torch.randn(1, 4, 8, 4)
    mask = None
    if problem == "mask":
        mask = torch.ones(1, 8)
    elif problem == "boundaries":
        kwargs["cu_seq_lens_k"] = torch.tensor([0, 4, 8], dtype=torch.int32)
    elif problem == "dtype":
        kwargs["cu_seq_lens_q"] = kwargs["cu_seq_lens_q"].long()
    else:
        kwargs["dropout"] = 0.1
    with pytest.raises(ValueError):
        packed_fa4_interface(sdpa, varlen)(None, q, q, q, mask, **kwargs)


def test_candidate_requires_version_and_fresh_validation():
    config = student_config()
    config["training"]["packed_attention_backend"] = "flash_attention_4"
    with pytest.raises(ValueError):
        validate_packed_training_config(config)
    config["training"]["packed_attention_version"] = "4.0.0b33"
    assert validate_packed_training_config(config)
    config["training"]["startup_validation_reference"] = "old_sdpa_receipt"
    with pytest.raises(ValueError, match="fresh startup"):
        validate_packed_training_config(config)


def test_matched_job_preserves_sources_and_only_changes_attention():
    config = yaml.safe_load(Path("experiments/b200_bf16_fa4/config.yaml").read_text())
    recipe = yaml.safe_load(Path(config["profile"]).read_text())["recipe"]
    source = {"selection_sha256": "selection", "soft_targets_sha256": "targets"}
    initial = deepcopy(source)
    control = benchmark_job(config, source, recipe, "sdpa")
    candidate = benchmark_job(config, source, recipe, "flash_attention_4")
    assert source == initial
    for job in [control, candidate]:
        assert all(job[k] == v for k, v in source.items())
        assert job["full_bf16_lora"] and job["sequence_packing"]
        assert job["micro_batch_size"] == 32 and not job["gradient_checkpointing"]
        assert job["max_steps"] == 20
    assert "startup_validation_reference" in control
    assert "startup_validation_reference" not in candidate


def test_explicit_learning_acceptance_is_bounded_and_preserves_strict_failure():
    from gleipnir.monitoring_training_command import training_command
    from gleipnir.packed_training_screen import PackingCanaryError, _accept_canary

    receipt = {
        "cases": [
            {
                "repeat_max_abs": 0.0,
                "perturb_max_abs": 0.0,
                "cross_input_grad_max_abs": 0.0,
                "own_input_grad_max_abs": 1.0,
            }
        ],
        "independent_loss": 0.9492289,
        "packed_loss": 0.9576337,
        "adapter_gradient_relative_l2": 0.0828666,
    }
    with pytest.raises(PackingCanaryError):
        _accept_canary(deepcopy(receipt), None)
    accepted = _accept_canary(deepcopy(receipt), 0.10)
    assert not accepted["passed"] and accepted["accepted_for_learning_comparison"]
    leaked = deepcopy(receipt)
    leaked["cases"][0]["cross_input_grad_max_abs"] = 1e-3
    with pytest.raises(PackingCanaryError):
        _accept_canary(leaked, 0.10)
    config = student_config()
    config["training"]["packing_learning_gradient_tolerance"] = 0.10
    assert validate_packed_training_config(config)
    config["training"]["packing_learning_gradient_tolerance"] = 0.151
    with pytest.raises(ValueError):
        validate_packed_training_config(config)
    screen = yaml.safe_load(
        Path("experiments/b200_bf16_fa4/accepted_config.yaml").read_text()
    )
    source = yaml.safe_load(Path(screen["profile"]).read_text())["recipe"]
    job = benchmark_job(
        screen,
        {
            "seed": 0,
            "student_rows": "rows.jsonl",
            "soft_targets": "targets.jsonl",
            "max_length": 29696,
            "rank": 128,
            "lora_alpha": 256,
            "learning_rate": 5e-5,
            "num_train_epochs": -1,
        },
        source,
        "flash_attention_4",
    )
    assert job["packing_learning_gradient_tolerance"] == 0.10
    assert (
        "++student.training.packing_learning_gradient_tolerance=0.1"
        in training_command(job)
    )
