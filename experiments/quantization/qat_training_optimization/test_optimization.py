"""Controls for unchanged STE gradients and privileged target identities."""

import json
from types import SimpleNamespace

import pytest
import torch

from experiments.quantization_recovery.test_recovery import (
    test_packed_mlp_recomputation_preserves_dense_control_gradients as dense_control,
)
from gleipnir.data.monitoring import file_hash
from gleipnir.training import quantization_recovery as reference
from gleipnir.training.distillation_targets import load_binary_targets
from gleipnir.training.qat_optimization import (
    LeanPackedMLP,
    native_fp8_linear,
    optimized_path,
)


def test_lean_backward_matches_dense_gradient_control(monkeypatch):
    monkeypatch.setattr(reference, "_PackedMLP", LeanPackedMLP)
    contractions = []

    class RecordingPatch:
        def setattr(self, module, name, value):
            if module is reference and name == "native_linear":
                original = value

                def record(a, b):
                    contractions.append((a, b))
                    return original(a, b)

                value = record
            monkeypatch.setattr(module, name, value)

    dense_control(RecordingPatch())
    # Two forward contractions, one gate/up reconstruction in backward.
    assert len(contractions) == 3


def test_scoped_path_restores_methods_and_parameters():
    group = SimpleNamespace(precision="fp8", forward=lambda x, i: x)
    model = SimpleNamespace(_gleipnir_qad_groups=[group])
    original_forward, original_mlp = group.forward, reference._PackedMLP
    with pytest.raises(RuntimeError), optimized_path(model, fp8=True):
        assert reference._PackedMLP is LeanPackedMLP
        assert group.forward is not original_forward
        raise RuntimeError("trial failure")
    assert group.forward is original_forward
    assert reference._PackedMLP is original_mlp


def test_native_fp8_preserves_fp32_accumulation_contract(monkeypatch):
    calls = []

    def gemm(a, b, **kwargs):
        calls.append(kwargs)
        return a.float() @ b.float()

    monkeypatch.setattr(torch, "_scaled_mm", gemm)
    a = torch.randn(5, 16).to(torch.float8_e4m3fn)
    b = torch.randn(32, 16).to(torch.float8_e4m3fn)
    assert torch.equal(native_fp8_linear(a, b), a.float() @ b.float().t())
    assert calls[0]["out_dtype"] == torch.float32
    assert calls[0]["use_fast_accum"] is False
    assert calls[0]["scale_a"].item() == calls[0]["scale_b"].item() == 1


@pytest.mark.parametrize("mutate", [False, True])
def test_shared_frozen_projection_matches_values_and_gradients(monkeypatch, mutate):
    torch.manual_seed(5)
    modules = (
        torch.nn.Linear(8, 7, bias=False).double(),
        torch.nn.Linear(8, 9, bias=False).double(),
    )
    for module in modules:
        module.weight.requires_grad_(False)
    group = reference.ProjectionGroup(modules, "fp4")
    group.prepare = lambda: None
    group.packed = torch.cat([m.weight for m in modules])
    group.decoded = group.packed
    group.operator_check = {"passed": True}
    contractions = []

    def linear(a, b):
        contractions.append(1)
        return torch.nn.functional.linear(a, b)

    monkeypatch.setattr(reference, "pack_activation", lambda x: x)
    monkeypatch.setattr(reference, "native_linear", linear)
    model = SimpleNamespace(_gleipnir_qad_groups=[group])
    x = torch.randn(1, 5, 8, dtype=torch.float64, requires_grad=True)
    with optimized_path(model):
        first = group.forward(x, 0)
        if mutate:
            with torch.no_grad():
                x.add_(0.125)
        second = group.forward(x, 1)
        assert group._shared_projection is None
    assert not hasattr(group, "_shared_projection")
    assert len(contractions) == (2 if mutate else 1)
    torch.testing.assert_close(second, modules[1](x), rtol=0, atol=0)
    if not mutate:
        observed = torch.cat((first, second), dim=-1)
        expected = torch.cat([m(x) for m in modules], dim=-1)
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        gradient = torch.randn_like(observed)
        actual = torch.autograd.grad(observed, x, gradient)[0]
        target = torch.autograd.grad(expected, x, gradient)[0]
        torch.testing.assert_close(actual, target, rtol=0, atol=0)


def cache_fixture(tmp_path):
    workload = [
        {"dataset": "source", "id": "one", "label": 0, "prompt_sha256": "a" * 64}
    ]
    teacher = {
        "dataset": "source",
        "index": "one",
        "label": 0,
        "teacher_model": "kimi",
        "rendered_prompt_sha256": "b" * 64,
        "soft_target": 0.2,
        "target_probs": {"negative": 0.8, "positive": 0.2},
    }
    path = tmp_path / "teacher.jsonl"
    path.write_text(json.dumps(teacher) + "\n")
    return workload, teacher, path


def test_privileged_cache_preserves_distinct_prompt_hashes(tmp_path):
    workload, _, path = cache_fixture(tmp_path)
    targets, receipt = load_binary_targets(
        workload,
        path,
        expected_sha256=file_hash(path),
        teacher_model="kimi",
        prompt_contract="privileged",
    )
    assert targets == [0.2]
    assert receipt["bindings"][0]["student_prompt_sha256"] == "a" * 64
    assert receipt["bindings"][0]["teacher_prompt_sha256"] == "b" * 64
    with pytest.raises(ValueError, match="same-prompt"):
        load_binary_targets(
            workload,
            path,
            expected_sha256=file_hash(path),
            teacher_model="kimi",
            prompt_contract="same",
        )


@pytest.mark.parametrize(
    "failure", ["duplicate", "membership", "label", "distribution", "checksum", "model"]
)
def test_cache_rejects_identity_or_distribution_drift(tmp_path, failure):
    workload, teacher, path = cache_fixture(tmp_path)
    if failure == "duplicate":
        workload *= 2
    elif failure == "membership":
        workload[0]["id"] = "other"
    elif failure == "label":
        teacher["label"] = 1
    elif failure == "distribution":
        teacher["soft_target"] = float("nan")
    elif failure == "model":
        teacher["teacher_model"] = "other"
    path.write_text(json.dumps(teacher) + "\n")
    with pytest.raises(ValueError):
        load_binary_targets(
            workload,
            path,
            expected_sha256="c" * 64 if failure == "checksum" else file_hash(path),
            teacher_model="kimi",
            prompt_contract="privileged",
        )
