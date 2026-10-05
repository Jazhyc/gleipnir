"""Focused adapter/state/gradient contract checks, independent of CUDA fusion."""

import pytest
import torch
from transformers import Qwen3_5TextConfig

from experiments.b200_mlp_gemm.probe import make_mlp, relative_l2
from experiments.b200_mlp_gemm.training_screen import accept_pilot
from gleipnir.cudnn_lora_mlp import cudnn_forward
from gleipnir.mlp_gemm import install_merged_mlp


def small_mlp():
    config = Qwen3_5TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["full_attention"],
    )
    return make_mlp(config, device="cpu")


def test_merged_preserves_state_and_nonzero_adapter_gradients():
    torch.manual_seed(41)
    m = small_mlp()
    parameters = [p for p in m.parameters() if p.requires_grad]
    original = m.forward
    original_state = {k: v.clone() for k, v in m.state_dict().items()}
    identities = [id(p) for p in m.parameters()]
    x = torch.randn(19, 32, dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = original(x)
    expected_grads = torch.autograd.grad(expected, (x, *parameters), dy)
    metadata = install_merged_mlp(m)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = m(x)
    actual_grads = torch.autograd.grad(actual, (x, *parameters), dy)
    assert metadata["extra_frozen_buffer_bytes"] == 2 * 64 * 32 * 2
    assert [id(p) for p in m.parameters()] == identities
    assert original_state.keys() == m.state_dict().keys()
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in original_state.items())
    assert torch.equal(actual, expected)
    assert len(actual_grads) == 7
    assert all(torch.isfinite(g).all() for g in actual_grads)
    assert all(
        relative_l2(a, b) <= 0.01
        for a, b in zip(actual_grads, expected_grads, strict=True)
    )


@pytest.mark.parametrize(
    "change", ["dropout", "master", "frozen", "multiple", "activation"]
)
def test_rejects_unsupported_contract_without_partial_mutation(change):
    m = small_mlp()
    if change == "dropout":
        m.up_proj.lora_dropout["default"] = torch.nn.Dropout(0.1)
    elif change == "master":
        m.up_proj.lora_A["default"].weight.data = m.up_proj.lora_A[
            "default"
        ].weight.bfloat16()
    elif change == "frozen":
        m.up_proj.base_layer.weight.requires_grad_(True)
    elif change == "multiple":
        m.up_proj.set_adapter(["default", "other"])
    else:
        m.config.hidden_act = "relu"
    with pytest.raises(ValueError):
        install_merged_mlp(m)
    assert not hasattr(m, "_gleipnir_gate_up")


def test_rejects_reinstallation():
    m = small_mlp()
    install_merged_mlp(m)
    with pytest.raises(ValueError, match="already installed"):
        install_merged_mlp(m)


def test_native_backward_contract_against_actual_peft(monkeypatch):
    # Replace only the GPU forward graph, leaving the custom backward real.
    import torch.nn.functional as F

    import gleipnir.cudnn_lora_mlp as native

    def fused(x, wg, wu, rg, ru):
        gate = F.linear(x, wg) + rg
        up = F.linear(x, wu) + ru
        return F.silu(gate) * up, gate, up

    monkeypatch.setattr(native, "_fused", fused)
    torch.manual_seed(41)
    m = small_mlp()
    parameters = [p for p in m.parameters() if p.requires_grad]
    x = torch.randn(19, 32, dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = m(x)
    expected_grads = torch.autograd.grad(expected, (x, *parameters), dy)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = cudnn_forward(m, x)
    actual_grads = torch.autograd.grad(actual, (x, *parameters), dy)
    assert torch.equal(actual, expected)
    assert all(
        relative_l2(a, b) <= 0.01
        for a, b in zip(actual_grads, expected_grads, strict=True)
    )


def pilot_receipt():
    return {
        "status": "complete",
        "variant": "merged_compiled",
        "timing_baseline": "compiled_peft",
        "full_backward_included": True,
        "shapes": [
            {
                "tokens": t,
                "finite": True,
                "output_relative_l2": 0.0,
                "gradient_relative_l2": [0.0] * 7,
                "row_isolation_relative_l2": 0.0,
                "step_ms": {"baseline": [10.0] * 10, "candidate": [8.0] * 10},
            }
            for t in (193, 4096, 16384)
        ],
    }


def test_training_selection_requires_compiled_backward_gain():
    receipt = pilot_receipt()
    accept_pilot(receipt)
    receipt["shapes"][2]["step_ms"]["candidate"] = [9.8] * 10
    with pytest.raises(ValueError, match="five-percent"):
        accept_pilot(receipt)


@pytest.mark.parametrize(
    "change",
    ["eager", "forward_only", "missing_gradient", "nan", "partial", "timing_nan"],
)
def test_training_selection_rejects_invalid_receipts(change):
    receipt = pilot_receipt()
    if change == "eager":
        receipt["timing_baseline"] = "eager_peft"
    elif change == "forward_only":
        receipt["full_backward_included"] = False
    elif change == "missing_gradient":
        receipt["shapes"][0]["gradient_relative_l2"].pop()
    elif change == "nan":
        receipt["shapes"][0]["row_isolation_relative_l2"] = float("nan")
    elif change == "partial":
        receipt["shapes"].pop()
    else:
        receipt["shapes"][0]["step_ms"]["candidate"][0] = float("nan")
    with pytest.raises(ValueError):
        accept_pilot(receipt)
