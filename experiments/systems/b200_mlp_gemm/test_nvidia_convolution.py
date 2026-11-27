"""Packing propagation, fail-closed routing and scoped restoration."""

from types import SimpleNamespace

import pytest
import torch

import gleipnir.nvidia_causal_conv1d as integration


class Qwen3_5GatedDeltaNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            8, 8, 4, groups=8, bias=False, dtype=torch.bfloat16
        )
        self.conv1d.requires_grad_(False)
        self.activation = "silu"
        self.causal_conv1d_fn = lambda **kw: kw["x"]

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        return self.causal_conv1d_fn(
            x=hidden_states,
            weight=self.conv1d.weight.squeeze(1),
            activation="silu",
            seq_idx=kwargs.get("seq_idx"),
        )


def backend(calls):
    def run(x, weight, bias, activation, *, cu_seqlens):
        calls.append(cu_seqlens)
        return x * 2

    return SimpleNamespace(
        _can_route_causal_conv1d_bulk=lambda *a: True,
        _get_causal_conv1d_last_route=lambda: "native-autograd",
        causal_conv1d=run,
        _compile_causal_conv1d_training_backend=lambda *a: None,
        _CAUSAL_CONV1D_TRAINING_CACHE_CAPACITY=64,
        _API_CACHE_CAPACITY=128,
        CausalConv1dBulkFwdSm100=type("Forward", (), {"compile": lambda self: None}),
    )


def test_scoped_adapter_preserves_offsets_gradients_and_disabled_boundary(monkeypatch):
    model = torch.nn.ModuleList([Qwen3_5GatedDeltaNet() for _ in range(24)])
    model[0].forward = torch.compiler.disable(model[0].forward)
    original = model[0].forward
    identities = [id(p) for p in model.parameters()]
    calls = []
    native = backend(calls)
    monkeypatch.setattr(integration.importlib, "import_module", lambda _: native)
    x = torch.randn(1, 8, 5, requires_grad=True)
    seq = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32)
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    with integration.nvidia_convolution_context(model) as receipt:
        assert native._CAUSAL_CONV1D_TRAINING_CACHE_CAPACITY == 256
        assert native._API_CACHE_CAPACITY == 256
        assert model[0].forward._torchdynamo_disable
        model[0](x, seq_idx=seq, cu_seq_lens_q=cu, cu_seq_lens_k=cu).sum().backward()
        assert calls == [cu] and receipt["stats"]["native-autograd"] == 1
        assert torch.equal(x.grad, torch.full_like(x, 2))
        assert [id(p) for p in model.parameters()] == identities
        with pytest.raises(ValueError, match="cumulative"):
            model[0](x, seq_idx=seq)
        with pytest.raises(ValueError, match="uncached"):
            model[0](x, cache_params=object())
        assert integration._BOUNDARIES.get() is None
    assert model[0].forward is original
    assert native._CAUSAL_CONV1D_TRAINING_CACHE_CAPACITY == 64
    assert native._API_CACHE_CAPACITY == 128
    assert all("forward" not in m.__dict__ for m in model[1:])
    assert torch.equal(model[0](x), x)


def test_exception_restores_forward_and_kernel(monkeypatch):
    model = torch.nn.ModuleList([Qwen3_5GatedDeltaNet() for _ in range(24)])
    original = model[0].causal_conv1d_fn
    native = backend([])
    compile_original = native._compile_causal_conv1d_training_backend
    monkeypatch.setattr(integration.importlib, "import_module", lambda _: native)
    with pytest.raises(RuntimeError, match="injected"):
        with integration.nvidia_convolution_context(model):
            raise RuntimeError("injected")
    assert model[0].causal_conv1d_fn is original
    assert native._compile_causal_conv1d_training_backend is compile_original
    assert all("forward" not in m.__dict__ for m in model)


def test_unsupported_native_route_fails_closed():
    native = backend([])
    native._can_route_causal_conv1d_bulk = lambda *a: False
    kernel = integration.convolution_kernel(native, {})
    token = integration._BOUNDARIES.set((None, None))
    try:
        with pytest.raises(ValueError, match="unsupported"):
            kernel(torch.ones(1, 8, 4), torch.ones(8, 4), activation="silu")
    finally:
        integration._BOUNDARIES.reset(token)


def test_dao_passthrough_preserves_reference_gradient():
    native = backend([])
    x = torch.ones(1, 8, 4, requires_grad=True)
    records = []
    kernel = integration.convolution_kernel(
        native,
        {},
        reference=lambda **kw: kw["x"] * 3,
        diagnostics=records,
        name="first",
        return_reference=True,
    )
    token = integration._BOUNDARIES.set((None, None))
    try:
        kernel(x, torch.ones(8, 4), activation="silu").sum().backward()
    finally:
        integration._BOUNDARIES.reset(token)
    assert torch.equal(x.grad, torch.full_like(x, 3))
    assert records[0]["module"] == "first"
    assert records[0]["relative_l2"] == pytest.approx(1 / 3)


def test_trainable_filters_are_rejected_before_installation(monkeypatch):
    model = torch.nn.ModuleList([Qwen3_5GatedDeltaNet() for _ in range(24)])
    model[-1].conv1d.weight.requires_grad_(True)
    native = backend([])
    monkeypatch.setattr(integration.importlib, "import_module", lambda _: native)
    with pytest.raises(ValueError, match="frozen"):
        with integration.nvidia_convolution_context(model):
            pass
    assert all("forward" not in m.__dict__ for m in model)
