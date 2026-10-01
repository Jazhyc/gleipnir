"""CPU checks of the opt-in kernel contract and numerical diagnostics."""

import pytest
import torch
import yaml

from experiments.fp4_stability.run import validate_config
from gleipnir.flashqla_training import make_flashqla_kernel, tensor_comparison


def test_wrapper_preserves_all_supported_arguments():
    calls = []

    def kernel(**kwargs):
        calls.append(kwargs)
        return "output", "state"

    wrapper = make_flashqla_kernel(kernel, auto_cp=False)
    tensors = [object() for _ in range(5)]
    state, lengths = object(), object()
    assert wrapper(
        *tensors,
        scale=0.125,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=lengths,
        state_v_first=True,
    ) == ("output", "state")
    assert list(calls[0].values())[:5] == tensors
    assert calls[0]["initial_state"] is state
    assert calls[0]["cu_seqlens"] is lengths
    assert calls[0]["scale"] == 0.125
    assert calls[0]["state_v_first"] is True
    assert calls[0]["output_final_state"] is True
    assert calls[0]["use_qk_l2norm_in_kernel"] is True
    assert calls[0]["auto_cp"] is False
    assert calls[0]["enable_fwd_cp_cache"] is True


def test_wrapper_rejects_unsupported_fla_gate_fusion():
    wrapper = make_flashqla_kernel(lambda **kwargs: None, auto_cp=True)
    with pytest.raises(TypeError, match="use_gate_in_kernel"):
        wrapper(1, 2, 3, 4, 5, use_gate_in_kernel=True)


def test_relative_l2_and_nonfinite():
    result = tensor_comparison(torch.tensor([3.0, 4.0]), torch.tensor([0.0, 4.0]))
    assert result["relative_l2"] == 0.75
    assert result["max_absolute_error"] == 3
    assert result["finite"]
    assert not tensor_comparison(torch.tensor([float("nan")]), torch.zeros(1))["finite"]


def test_zero_reference_does_not_hide_error():
    assert tensor_comparison(torch.ones(1), torch.zeros(1))["relative_l2"] == 1e12


def test_auto_partitioning_requires_flashqla():
    from pathlib import Path

    config = yaml.safe_load(
        (Path(__file__).parent / "row_flashqla_timing.yaml").read_text()
    )
    validate_config(config)
    config.update(gated_delta_backend="fla", flashqla_auto_cp=True)
    with pytest.raises(ValueError, match="partitioning requires FlashQLA"):
        validate_config(config)


@pytest.mark.parametrize("factor,passed", [(1.001, True), (2.0, False)])
def test_model_gate_preserves_or_restores_kernel(monkeypatch, factor, passed):
    import gleipnir.flashqla_training as backend

    def reference(q, *args, **kwargs):
        return q, None

    reference.__module__ = "fla.ops.gated_delta_rule"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(24)])
            for layer in self.layers:
                layer.chunk_gated_delta_rule = reference

        def forward(self, value):
            q = value * self.weight
            outputs = [
                layer.chunk_gated_delta_rule(q, q, q, q, q)[0] for layer in self.layers
            ]
            return torch.stack(outputs).mean()

    monkeypatch.setattr(
        backend, "load_flashqla", lambda: (lambda q, **kw: (q * factor, None), {})
    )
    model = Model()
    result = backend.install_with_model_canary(
        model, [torch.tensor(1.0), torch.tensor(2.0)], model, auto_cp=False
    )
    assert result["passed"] is passed
    assert model.training
    assert model.weight.item() == 1.0
    assert model.weight.grad is None
    assert all(
        (layer.chunk_gated_delta_rule is reference) is (not passed)
        for layer in model.layers
    )
