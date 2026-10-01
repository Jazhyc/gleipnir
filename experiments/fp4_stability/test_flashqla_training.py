"""CPU checks of the opt-in kernel contract and numerical diagnostics."""

import pytest
import torch
import yaml

from experiments.fp4_stability.run import validate_config
from gleipnir.flashqla_training import (
    make_bf16_boundary,
    make_flashqla_kernel,
    tensor_comparison,
)


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


def test_boundary_keeps_gates_fp32_and_backpropagates():
    observed = []

    def kernel(q, k, v, g, beta, **kwargs):
        observed.append([x.dtype for x in (q, k, v, g, beta)])
        return q + k + v + beta, None

    tensors = [torch.ones(2, requires_grad=True) for _ in range(5)]
    boundary = make_bf16_boundary(kernel)
    output, state = boundary(*tensors, output_final_state=False)
    assert output.dtype == torch.float32 and state is None
    assert observed == [[torch.bfloat16] * 3 + [torch.float32, torch.bfloat16]]
    output.sum().backward()
    assert tensors[0].grad.dtype == torch.float32
    assert torch.equal(tensors[0].grad, torch.ones(2))
    assert boundary.input_dtypes == [["torch.float32"] * 5]


def test_auto_partitioning_requires_flashqla():
    from pathlib import Path

    config = yaml.safe_load(
        (Path(__file__).parent / "row_flashqla_timing.yaml").read_text()
    )
    validate_config(config)
    config.update(gated_delta_backend="fla", flashqla_auto_cp=True)
    with pytest.raises(ValueError, match="partitioning requires FlashQLA"):
        validate_config(config)


@pytest.mark.parametrize(
    "policy,dtype",
    [
        ("bf16_fp32_gates_norm", torch.bfloat16),
        ("fp16_fp32_gates_norm", torch.float16),
    ],
)
def test_precise_boundary_preserves_gates_and_normalizes_before_cast(
    monkeypatch, policy, dtype
):
    import gleipnir.flashqla_training as helper

    normalizer_inputs, calls = [], []

    def normalize(x):
        normalizer_inputs.append(x.dtype)
        return x / (x.square().sum(-1, keepdim=True) + 1e-6).sqrt()

    monkeypatch.setattr(helper, "_normalize_qk_fp32", normalize)

    def kernel(q, k, v, g, beta, **kwargs):
        calls.append(([x.dtype for x in (q, k, v, g, beta)], kwargs))
        assert not kwargs["use_qk_l2norm_in_kernel"]
        assert torch.allclose(q.float().square().sum(-1), torch.ones(2), atol=0.01)
        return q + k + v + beta[..., None] + g[..., None], None

    q, k, v = [
        torch.tensor([[3.0, 4.0], [4.0, 3.0]], requires_grad=True) for _ in range(3)
    ]
    g, beta = [
        torch.tensor([0.1234567, 0.8765432], requires_grad=True) for _ in range(2)
    ]
    output, _ = helper.make_precision_boundary(kernel, policy=policy)(
        q, k, v, g, beta, use_qk_l2norm_in_kernel=True
    )
    assert normalizer_inputs == [torch.float32, torch.float32]
    assert calls[0][0] == [dtype] * 3 + [torch.float32] * 2
    assert output.dtype == torch.float32
    output.sum().backward()
    assert all(
        x.grad is not None and torch.isfinite(x.grad).all() for x in [q, k, v, g, beta]
    )
    assert torch.equal(beta.grad, torch.full_like(beta, 2))


def test_boundary_policy_rejects_unknown_or_inactive_policy():
    from pathlib import Path

    from gleipnir.flashqla_training import make_precision_boundary

    with pytest.raises(ValueError, match="unknown GDN"):
        make_precision_boundary(lambda: None, policy="fp4")
    config = yaml.safe_load(
        (Path(__file__).parent / "nf4_flashqla_timing.yaml").read_text()
    )
    with pytest.raises(ValueError, match="inactive GDN"):
        validate_config({**config, "gated_delta_boundary_policy": "fp4"})
    with pytest.raises(ValueError, match="inactive GDN"):
        validate_config(
            {
                **config,
                "gated_delta_boundary_policy": "fp16_fp32_gates_norm",
                "gated_delta_bf16_boundary": False,
            }
        )


@pytest.mark.parametrize("factor,passed", [(1.001, True), (2.0, False)])
@pytest.mark.parametrize("bf16_boundary", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_model_gate_preserves_or_restores_kernel(
    monkeypatch, factor, passed, bf16_boundary, partial
):
    import gleipnir.flashqla_training as backend

    def reference(q, *args, **kwargs):
        return q, None

    reference.__module__ = "fla.ops.gated_delta_rule"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(24)])
            for index, layer in zip(
                [i for i in range(32) if i % 4 != 3], self.layers, strict=True
            ):
                layer.chunk_gated_delta_rule = reference
                layer.layer_idx = index

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
        model,
        [torch.tensor(1.0), torch.tensor(2.0)],
        model,
        auto_cp=False,
        bf16_boundary=bf16_boundary,
        layer_indices=[29, 30] if partial else None,
    )
    assert result["passed"] is passed
    assert model.training
    assert model.weight.item() == 1.0
    assert model.weight.grad is None
    assert result["replaced_layers"] == (2 if partial else 24)
    for layer in model.layers:
        changed = passed and (not partial or layer.layer_idx in [29, 30])
        assert (layer.chunk_gated_delta_rule is reference) is (not changed)
    if bf16_boundary and not passed:
        assert "failure_diagnostic_error" not in result
        # BF16 casts round the 1/24 gradient contributions in this toy mean.
        assert result["fla_bf16_failure_control"]["gradient_relative_l2"] < 0.005
        assert len(result["shadow_outputs_on_original_path"]) == (4 if partial else 48)
        assert all(
            row["fla_bf16"]["relative_l2"] == 0
            for row in result["shadow_outputs_on_original_path"]
        )


@pytest.mark.parametrize("indices", [[], [3], [32], [30, 29], [True], [29, 29]])
def test_layer_selection_rejects_invalid_decoder_layers(indices):
    from pathlib import Path

    config = yaml.safe_load(
        (
            Path(__file__).parent / "nf4_flashqla_bf16_precise_diagnostic.yaml"
        ).read_text()
    )
    with pytest.raises(ValueError, match="decoder-layer"):
        validate_config({**config, "flashqla_layer_indices": indices})
    validate_config({**config, "flashqla_layer_sweep": [[29, 30], [30]]})
    with pytest.raises(ValueError, match="diagnostic-only"):
        validate_config(
            {
                **config,
                "diagnostics_only": False,
                "flashqla_layer_sweep": [[29, 30], [30]],
            }
        )
