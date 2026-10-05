"""Real custom-op autograd/compiler wiring with mocked native CUDA arithmetic."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import gleipnir.cudnn_fp4_mlp as native
from experiments.b200_mlp_gemm.probe import make_mlp, relative_l2


def module():
    return make_mlp(
        SimpleNamespace(hidden_size=64, intermediate_size=128, hidden_act="silu"),
        device="cpu",
    )


@pytest.fixture
def decoded_native(monkeypatch):
    calls = []

    def kernel(x, weight, other, backward):
        calls.append(backward)
        w = weight if other is None else torch.cat((weight, other))
        return F.linear(x, w.t() if backward else w)

    monkeypatch.setattr(native, "_native_linear", kernel)
    return calls


def test_registered_backward_preserves_all_master_gradients(decoded_native):
    torch.manual_seed(41)
    m = module()
    original = m.forward
    masters = [p for p in m.parameters() if p.requires_grad]
    identities = [id(p) for p in m.parameters()]
    state = tuple(m.state_dict())
    x = torch.randn(1, 19, 64, dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        ref = original(x)
        ref_grads = torch.autograd.grad(ref, (x, *masters), dy)
    native.install_fp4_mlp(m)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = m(x)
        grads = torch.autograd.grad(y, (x, *masters), dy)
    assert torch.equal(ref, y)
    assert max(relative_l2(a, b) for a, b in zip(grads, ref_grads, strict=True)) < 0.01
    assert decoded_native.count(False) == 2 and decoded_native.count(True) == 2
    assert [id(p) for p in m.parameters()] == identities and tuple(
        m.state_dict()
    ) == state
    assert len(masters) == 6 and all(p.dtype == torch.float32 for p in masters)
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_fake_and_fullgraph_backward_updates_live_masters(decoded_native):
    torch.manual_seed(41)
    m = module()
    native.install_fp4_mlp(m)
    compiled = torch.compile(m, backend="aot_eager", fullgraph=True, dynamic=True)
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.01)
    initial = {n: p.detach().clone() for n, p in m.named_parameters()}
    x = torch.randn(19, 64, dtype=torch.bfloat16, requires_grad=True)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            compiled(x).float().square().mean().backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in m.parameters()
            if p.requires_grad
        )
        opt.step()
    assert all(
        torch.equal(p, initial[n])
        for n, p in m.named_parameters()
        if not p.requires_grad
    )
    assert any(
        not torch.equal(p, initial[n])
        for n, p in m.named_parameters()
        if p.requires_grad
    )


def test_install_rejects_unsupported_before_partial_mutation():
    m = module()
    m.up_proj.lora_dropout["default"] = torch.nn.Dropout(0.1)
    with pytest.raises(ValueError, match="zero dropout"):
        native.install_fp4_mlp(m)
    assert not hasattr(m, "_gleipnir_fp4_installed")


def test_missing_gpu_fails_closed():
    with pytest.raises(ValueError, match="CUDA BF16"):
        native.fp4_frozen_linear(
            torch.zeros(19, 64, dtype=torch.bfloat16),
            torch.zeros(128, 64, dtype=torch.bfloat16),
            None,
        )


def integrated_receipt():
    return {
        "status": "complete",
        "variant": "fp4_integrated",
        "timing_baseline": "compiled_peft",
        "full_backward_included": True,
        "shapes": [
            {
                "tokens": t,
                "finite": True,
                "oracle_output_relative_l2": 0.003,
                "oracle_gradient_relative_l2": [0.004] * 7,
                "row_isolation_relative_l2": 0.0,
                "graph_copy_included": True,
                "changed_input_master_replay_relative_l2": [0.0] * 8,
                "graph_timing": {
                    "samples_ms": {"baseline": [10.0] * 10, "candidate": [8.0] * 10}
                },
            }
            for t in (193, 4096, 16384)
        ],
    }


@pytest.mark.parametrize(
    "failure", [None, "partial", "oracle", "replay", "timing", "gain"]
)
def test_model_selection_requires_complete_live_fp4_graph_evidence(failure):
    from experiments.b200_mlp_gemm.training_screen import accept_integrated_pilot

    r = integrated_receipt()
    if failure == "partial":
        r["shapes"].pop()
    elif failure == "oracle":
        r["shapes"][0]["oracle_gradient_relative_l2"][0] = float("nan")
    elif failure == "replay":
        r["shapes"][0]["changed_input_master_replay_relative_l2"].pop()
    elif failure == "timing":
        r["shapes"][0]["graph_timing"]["samples_ms"]["candidate"].pop()
    elif failure == "gain":
        r["shapes"][2]["graph_timing"]["samples_ms"]["candidate"] = [9.8] * 10
    if failure is None:
        accept_integrated_pilot(r)
    else:
        with pytest.raises(ValueError):
            accept_integrated_pilot(r)
