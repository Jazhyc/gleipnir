"""Verify bounded screen controls and restoration without loading a GPU model."""

from types import SimpleNamespace

import pytest
import torch

import gleipnir.packed_training_screen as screen


def test_layer_diagnostics_preserve_sequence_order_and_failure_receipt():
    reference = {"layer": [torch.ones(1, 3, 2), torch.ones(1, 5, 2)]}
    candidate = {"layer": [torch.ones(1, 8, 2)]}
    assert screen._layer_comparison(reference, candidate)[0]["max_absolute"] == 0
    candidate["layer"][0][:, 3:] += 1
    result = screen._layer_comparison(reference, candidate)
    assert result[0]["max_absolute"] == 1
    error = screen.PackingCanaryError("failed parity", {"layers": result})
    assert error.receipt["layers"] == result


@pytest.mark.parametrize("fail", [False, True])
def test_conditions_share_initial_weights_and_restore_bindings(
    monkeypatch, tmp_path, fail
):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(2)
    original_forward = model.forward

    def kernel():
        pass

    def convolve():
        pass

    convolve.__module__ = "causal_conv1d.causal_conv1d_interface"
    model.chunk_gated_delta_rule = kernel
    model.causal_conv1d_fn = convolve
    calls = []
    capture_modes = []
    monkeypatch.setattr(
        screen,
        "packing_isolation_canary",
        lambda *args, capture_layers: capture_modes.append(capture_layers),
    )

    def run(**options):
        condition = options["output"].name
        assert model.weight.item() == 2
        assert model.chunk_gated_delta_rule is kernel
        assert model.forward == original_forward
        assert options["packing_canary"] is not None
        options["packing_canary"](compiled=False)
        options["packing_canary"](compiled=True)
        if condition == "packed":
            assert options["partition_strategy"]([5, 3, 2]) == [[0, 1], [2]]
            assert options["collator"] is screen.collate_packed_monitoring
        calls.append(condition)
        model.weight.data.fill_(9)
        model.forward = lambda: None
        model.chunk_gated_delta_rule = lambda: None
        if fail:
            raise ValueError("gate failure")
        return {
            "timing_summary": {"mean_step_seconds": 2 if condition == "padded" else 1}
        }

    monkeypatch.setattr(screen, "run_precision_training_screen", run)
    options = dict(
        model=model,
        output=tmp_path,
        features=[],
        collator=object(),
        loss_forward=object(),
        optimizer_factory=lambda: None,
        policy=SimpleNamespace(max_padded_tokens=8),
        steps=10,
        metadata={"quantization": {"full_bf16_lora": {"verified": True}}},
        gated_delta_backend="flashqla",
        flashqla_auto_cp=False,
        ten_step_learning_comparison=True,
    )
    if fail:
        with pytest.raises(ValueError, match="gate failure"):
            screen.run_packed_training_screen(**options)
        assert calls == ["padded"]
        assert capture_modes == [True, False]
    else:
        report = screen.run_packed_training_screen(**options)
        assert calls == ["padded", "packed"]
        assert capture_modes == [True, False, True, False]
        assert report["step_time_reduction_fraction"] == 0.5
    assert model.weight.item() == 2
    assert model.forward == original_forward
    assert model.chunk_gated_delta_rule is kernel
