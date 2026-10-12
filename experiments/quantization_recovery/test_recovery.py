"""CPU controls for the new merged-weight and low-rank STE contracts."""

from types import SimpleNamespace

import pytest
import torch

from experiments.quantization_recovery.train import binary_kl
from gleipnir.training.quantization_recovery import (
    _ExactForward,
    _LowRankSTE,
    effective_weight,
    mxfp8_ste,
)


def test_packed_mlp_recomputation_preserves_dense_control_gradients(monkeypatch):
    from gleipnir.kernels.fp4.cudnn_fp4_gemm import PackedNvfp4
    from gleipnir.serving.fp4 import fusion
    from gleipnir.training import quantization_recovery as implementation

    torch.manual_seed(4)
    x = torch.randn(1, 5, 7, dtype=torch.float64, requires_grad=True)
    shapes = ((11, 7), (11, 7), (7, 11))
    modules, weights, masters = [], [], []
    for n, k in shapes:
        a = torch.randn(3, k, dtype=torch.float64, requires_grad=True)
        b = torch.randn(n, 3, dtype=torch.float64, requires_grad=True)
        base = torch.randn(n, k, dtype=torch.float64)
        modules.append(
            SimpleNamespace(
                active_adapters=["default"],
                lora_A={"default": SimpleNamespace(weight=a)},
                lora_B={"default": SimpleNamespace(weight=b)},
                scaling={"default": 2.0},
            )
        )
        weights.append(base + 2 * b @ a)
        masters.extend([a, b])

    def packed(tensor):
        return PackedNvfp4(tensor.detach(), torch.ones(1), torch.ones(1))

    pair = SimpleNamespace(
        prepare=lambda: None,
        modules=modules[:2],
        widths=[11, 11],
        packed=packed(torch.cat(weights[:2])),
        decoded=torch.cat(weights[:2]).detach(),
    )
    down = SimpleNamespace(
        prepare=lambda: None,
        modules=modules[2:],
        packed=packed(weights[2]),
        decoded=weights[2].detach(),
    )
    monkeypatch.setattr(implementation, "pack_activation", packed)
    monkeypatch.setattr(implementation, "decode_fp4", lambda operand: operand.codes)
    monkeypatch.setattr(
        implementation,
        "native_linear",
        lambda a, b: torch.nn.functional.linear(a.codes, b.codes),
    )
    monkeypatch.setattr(
        fusion,
        "silu_pack",
        lambda tensor: packed(
            torch.nn.functional.silu(tensor[:, :11]) * tensor[:, 11:]
        ),
    )
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(x, weights[0]))
        * torch.nn.functional.linear(x, weights[1]),
        weights[2],
    )
    observed = implementation._PackedMLP.apply(x, *masters, pair, down, None)
    torch.testing.assert_close(observed, expected)
    grad = torch.randn_like(expected)
    expected_grad = torch.autograd.grad(expected, (x, *masters), grad)
    observed_grad = torch.autograd.grad(observed, (x, *masters), grad)
    for actual, reference in zip(observed_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, reference)


def test_low_rank_ste_matches_dense_surrogate_gradients():
    torch.manual_seed(2)
    x = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    a = torch.randn(3, 7, dtype=torch.float64, requires_grad=True)
    b = torch.randn(11, 3, dtype=torch.float64, requires_grad=True)
    base = torch.randn(11, 7, dtype=torch.float64)
    w = base + 2 * b.detach() @ a.detach()
    g = torch.randn(5, 11, dtype=torch.float64)
    expected = torch.nn.functional.linear(x, base + 2 * b @ a)
    expected_grad = torch.autograd.grad(expected, (x, a, b), g)
    native = torch.randn_like(expected)
    observed = _LowRankSTE.apply(x, a, b, native, x.detach(), w, 2.0)
    assert torch.equal(observed, native)
    observed_grad = torch.autograd.grad(observed, (x, a, b), g)
    for actual, reference in zip(observed_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, reference)


def test_effective_weight_accumulates_before_bf16_rounding():
    base = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    a = torch.tensor([[0.002, 0.004]], dtype=torch.float32)
    b = torch.tensor([[1.0]], dtype=torch.float32)
    module = SimpleNamespace(
        base_layer=SimpleNamespace(weight=base),
        lora_A={"default": SimpleNamespace(weight=a)},
        lora_B={"default": SimpleNamespace(weight=b)},
        active_adapters=["default"],
        merged=False,
        disable_adapters=False,
        use_dora={"default": False},
        scaling={"default": 2.0},
    )
    assert torch.equal(
        effective_weight(module), (base.float() + 2 * b @ a).to(torch.bfloat16)
    )
    module.merged = True
    with pytest.raises(ValueError, match="unmerged"):
        effective_weight(module)


@pytest.mark.parametrize("column", [False, True])
def test_mxfp8_rounding_ste_handles_partial_groups(column):
    torch.manual_seed(3)
    x = torch.randn(33, 4, 256, requires_grad=True)
    y = mxfp8_ste(x, column=column)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    if column:
        padded = torch.nn.functional.pad(x.detach(), (0, 0, 0, 0, 0, 31))
        groups = padded.transpose(0, 2).reshape(256, 4, 2, 32)
        scales = torch.pow(
            2.0, torch.ceil(torch.log2(groups.abs().amax(-1, keepdim=True) / 448))
        )
        expected = (
            ((groups / scales).to(torch.float8_e4m3fn).float() * scales)
            .reshape(256, 4, 64)
            .transpose(0, 2)[:33]
        )
        assert torch.equal(y, expected)


def test_exact_native_forward_avoids_subtraction_rounding():
    surrogate = torch.tensor([1e5], requires_grad=True)
    native = torch.tensor([0.001])
    value = _ExactForward.apply(surrogate, native)
    assert torch.equal(value, native)
    value.sum().backward()
    assert surrogate.grad.item() == 1


def test_binary_kl_is_zero_at_teacher_and_ce_has_same_gradient():
    p = torch.tensor([0.001, 0.2, 0.8, 0.999])
    logits = torch.stack(((1 - p).log(), p.log()), -1).requires_grad_()
    loss = binary_kl(logits, p)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)
    gradient = torch.autograd.grad(loss.sum(), logits, retain_graph=True)[0]
    ce = -(torch.stack((1 - p, p), -1) * logits.log_softmax(-1)).sum()
    torch.testing.assert_close(gradient, torch.autograd.grad(ce, logits)[0])
