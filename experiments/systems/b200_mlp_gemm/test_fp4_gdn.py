"""GDN installation must preserve the recurrent shell, parameters and restore."""

import pytest
import torch
import torch.nn.functional as F

import gleipnir.cudnn_fp4_gdn as integration


class Qwen3_5GatedDeltaNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("in_proj_qkv", "in_proj_z", "out_proj"):
            setattr(
                self, name, torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
            )
        self.in_proj_a = torch.nn.Linear(64, 1, bias=False, dtype=torch.bfloat16)
        self.in_proj_b = torch.nn.Linear(64, 1, bias=False, dtype=torch.bfloat16)
        self.requires_grad_(False)

    def forward(self, hidden_states):
        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        z = self.in_proj_z(hidden_states)
        a = self.in_proj_a(hidden_states)
        b = self.in_proj_b(hidden_states)
        recurrence = mixed_qkv.transpose(1, 2).tanh() * (a + b).sigmoid()
        return self.out_proj(recurrence * F.silu(z))


@pytest.mark.parametrize("merged", [False, True])
def test_intervention_preserves_shell_gradient_and_parameter_identity(
    monkeypatch, merged
):
    modules = torch.nn.ModuleList([Qwen3_5GatedDeltaNet() for _ in range(24)])
    calls = []

    def linear(x, weight, other):
        calls.append(other is not None)
        return F.linear(x, weight if other is None else torch.cat((weight, other)))

    monkeypatch.setattr(integration, "fp4_epilogue_linear", linear)
    identities = [id(p) for p in modules.parameters()]
    x = torch.randn(1, 3, 64, dtype=torch.bfloat16, requires_grad=True)
    reference = modules[0](x)
    gradient = torch.autograd.grad(reference.sum(), x)[0]
    with integration.gdn_fp4_context(modules, merged_inputs=merged) as receipt:
        actual = modules[0](x)
        actual_gradient = torch.autograd.grad(actual.sum(), x)[0]
        assert torch.equal(actual, reference)
        torch.testing.assert_close(actual_gradient, gradient, rtol=0.04, atol=0.004)
        assert calls == ([True, False] if merged else [False, False, False])
        assert receipt["shared_input_packing"] is merged
        assert [id(p) for p in modules.parameters()] == identities
        assert "forward" not in modules[0].in_proj_a.__dict__
        assert "forward" not in modules[0].in_proj_b.__dict__
    assert all("forward" not in m.__dict__ for m in modules.modules())
    assert torch.equal(modules[0](x), reference)


def test_intervention_restores_after_exception():
    modules = torch.nn.ModuleList([Qwen3_5GatedDeltaNet() for _ in range(24)])
    with pytest.raises(RuntimeError, match="injected"):
        with integration.gdn_fp4_context(modules):
            raise RuntimeError("injected")
    assert all("forward" not in m.__dict__ for m in modules.modules())


def test_intervention_rejects_trainable_projections_before_installing():
    modules = torch.nn.ModuleList([Qwen3_5GatedDeltaNet() for _ in range(24)])
    modules[-1].out_proj.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen biasless"):
        with integration.gdn_fp4_context(modules):
            pass
    assert all("forward" not in m.__dict__ for m in modules.modules())


def changed_forward(self, hidden_states):
    return self.in_proj_qkv(hidden_states)


def test_source_transform_fails_closed_on_changed_projection_contract():
    with pytest.raises(ValueError, match="assignments changed"):
        integration.merged_gdn_forward(changed_forward)


def test_merged_forward_preserves_disabled_boundary(monkeypatch):
    modules = torch.nn.ModuleList([Qwen3_5GatedDeltaNet() for _ in range(24)])
    for module in modules:
        module.forward = torch.compiler.disable(module.forward)
    original = modules[0].forward
    monkeypatch.setattr(integration, "fp4_epilogue_linear", lambda x, weight, other: x)
    with integration.gdn_fp4_context(modules):
        assert modules[0].forward._torchdynamo_disable
    assert modules[0].forward is original


def test_actual_transformers_hook_wrapper_is_preserved():
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5GatedDeltaNet as ActualGDN,
    )

    function, receipt = integration.merged_gdn_forward(ActualGDN.forward)
    assert function is not ActualGDN.forward
    assert receipt["accelerate_hook_child"] == "conv1d"
