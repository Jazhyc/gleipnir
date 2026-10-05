"""Preserve head grouping, hooks, disabled boundaries and scoped restoration."""

import inspect
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from gleipnir.grouped_gdn import (
    grouped_gdn_context,
    grouped_gdn_forward,
    grouped_normalization_configs,
)


class Qwen3_5GatedDeltaNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_v_heads, self.num_k_heads = 4, 2
        self.weight = torch.nn.Parameter(torch.randn(2, 8))
        self.calls = []

    def forward(self, hidden_states, cache_params=None):
        query = hidden_states * self.weight
        key = hidden_states + self.weight
        value = torch.ones((*hidden_states.shape[:2], self.num_v_heads, 8))
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        return self.kernel(query, key, value)

    def kernel(self, q, k, v):
        self.calls.append((q.shape[2], k.shape[2], v.shape[2]))
        q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
        if q.shape[2] < v.shape[2]:
            q = q.repeat_interleave(v.shape[2] // q.shape[2], dim=2)
            k = k.repeat_interleave(v.shape[2] // k.shape[2], dim=2)
        return (q * k) * v


@pytest.mark.parametrize("disabled", [False, True])
def test_grouping_preserves_output_gradients_parameters_and_restoration(disabled):
    model = torch.nn.ModuleList([Qwen3_5GatedDeltaNet() for _ in range(24)])
    if disabled:
        for m in model:
            m.forward = torch.compiler.disable(m.forward)
    originals = [m.forward for m in model]
    identities = [id(p) for p in model.parameters()]
    x = torch.randn(1, 3, 2, 8, requires_grad=True)
    reference = model[0](x)
    reference_grad = torch.autograd.grad(reference.sum(), (x, model[0].weight))
    with grouped_gdn_context(model) as receipt:
        actual = model[0](x)
        actual_grad = torch.autograd.grad(actual.sum(), (x, model[0].weight))
        assert torch.equal(reference, actual)
        for a, b in zip(actual_grad, reference_grad, strict=True):
            torch.testing.assert_close(a, b)
        assert model[0].calls == [(4, 4, 4), (2, 2, 4)]
        assert receipt["query_key_heads"] == 2
        assert receipt["value_heads"] == 4
        assert [id(p) for p in model.parameters()] == identities
        assert (
            bool(getattr(model[0].forward, "_torchdynamo_disable", False)) == disabled
        )
        with pytest.raises(ValueError, match="cache-free"):
            model[0](x, cache_params=object())
    for m, original in zip(model, originals, strict=True):
        if disabled:
            assert m.forward is original
        else:
            assert "forward" not in m.__dict__


def test_restore_after_exception():
    model = torch.nn.ModuleList([Qwen3_5GatedDeltaNet() for _ in range(24)])
    with pytest.raises(RuntimeError, match="injected"):
        with grouped_gdn_context(model):
            raise RuntimeError("injected")
    assert all("forward" not in m.__dict__ for m in model)


@pytest.mark.parametrize("fail", [False, True])
def test_matched_normalization_config_and_function_restore(monkeypatch, fail):
    import gleipnir.flashqla_training as boundary

    key = (128, 1, "torch.float32", "torch.float32", "torch.float32")
    source = (128, 2, *key[2:])
    cache = {key: "grouped", source: "expanded"}
    autotuner = SimpleNamespace(cache=cache)
    monkeypatch.setitem(
        sys.modules,
        "fla.modules.l2norm",
        SimpleNamespace(l2norm_fwd_kernel=autotuner),
    )
    x = SimpleNamespace(shape=(1, 4096, 16, 128), numel=lambda: 4096 * 16 * 128)

    def original(actual):
        assert actual is x
        assert cache[key] == "expanded"
        if fail:
            raise RuntimeError("injected")
        return "normalized"

    monkeypatch.setattr(boundary, "_normalize_qk_fp32", original)
    with grouped_normalization_configs() as configs:
        if fail:
            with pytest.raises(RuntimeError, match="injected"):
                boundary._normalize_qk_fp32(x)
        else:
            assert boundary._normalize_qk_fp32(x) == "normalized"
        assert cache == {key: "grouped", source: "expanded"}
        assert configs == [{"grouped_nb": 1, "expanded_nb": 2, "config": "expanded"}]
    assert boundary._normalize_qk_fp32 is original


def test_actual_transformers_hook_preserved():
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5GatedDeltaNet as ActualGDN,
    )

    function, receipt = grouped_gdn_forward(ActualGDN.forward)
    assert function is not ActualGDN.forward
    assert receipt["accelerate_hook_child"] == "conv1d"
    assert inspect.getclosurevars(function).nonlocals["child_module_name"] == "conv1d"


def changed_forward(self, hidden_states):
    return hidden_states


def test_changed_forward_rejected():
    with pytest.raises(ValueError, match="replication block changed"):
        grouped_gdn_forward(changed_forward)


@pytest.mark.parametrize("count,heads", [(23, 4), (24, 3), (24, 2)])
def test_unsupported_layers_or_heads_rejected_before_install(count, heads):
    model = torch.nn.ModuleList([Qwen3_5GatedDeltaNet() for _ in range(count)])
    model[-1].num_v_heads = heads
    with pytest.raises(ValueError, match="all 24|divisible grouped"):
        with grouped_gdn_context(model):
            pass
    assert all("forward" not in m.__dict__ for m in model)
