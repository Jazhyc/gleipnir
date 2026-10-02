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


@pytest.mark.parametrize("packing_only", [False, True, None])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("learning_tolerance", [None, 0.15])
@pytest.mark.parametrize("cache_limit", [None, 64])
def test_conditions_share_initial_weights_and_restore_bindings(
    monkeypatch, tmp_path, fail, learning_tolerance, cache_limit, packing_only
):
    original_cache_limit = torch._dynamo.config.recompile_limit
    original_fail_on_limit = torch._dynamo.config.fail_on_recompile_limit_hit
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
        lambda *args, capture_layers, learning_tolerance: capture_modes.append(
            (capture_layers, learning_tolerance)
        ),
    )

    def run(**options):
        assert torch._dynamo.config.recompile_limit == (
            cache_limit or original_cache_limit
        )
        assert torch._dynamo.config.fail_on_recompile_limit_hit == (
            True if cache_limit is not None else original_fail_on_limit
        )
        condition = options["output"].name
        assert model.weight.item() == 2
        assert model.chunk_gated_delta_rule is kernel
        assert model.forward == original_forward
        assert options["packing_canary"] is not None
        assert options["common_probe_collator"] is options_original_collator
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
    options_original_collator = object()
    options = dict(
        model=model,
        output=tmp_path,
        features=[],
        collator=options_original_collator,
        loss_forward=object(),
        optimizer_factory=lambda: None,
        policy=SimpleNamespace(max_padded_tokens=8),
        steps=10,
        metadata={
            "quantization": {"full_bf16_lora": {"verified": True}},
            "packing_learning_gradient_tolerance": learning_tolerance,
            "packing_compile_cache_limit": cache_limit,
        },
        gated_delta_backend="flashqla",
        flashqla_auto_cp=False,
        ten_step_learning_comparison=True,
    )
    if packing_only is not None:
        options["metadata"]["packing_only"] = packing_only
    conditions = ["padded", "packed"] if packing_only is False else ["packed"]
    if fail:
        with pytest.raises(ValueError, match="gate failure"):
            screen.run_packed_training_screen(**options)
        assert calls == conditions[:1]
        assert capture_modes == [
            (True, learning_tolerance),
            (False, learning_tolerance),
        ]
    else:
        report = screen.run_packed_training_screen(**options)
        assert calls == conditions
        assert capture_modes == [
            (True, learning_tolerance),
            (False, learning_tolerance),
        ] * len(conditions)
        assert report["learning_gradient_tolerance"] == learning_tolerance
        assert report["packing_only"] is (packing_only is not False)
        if packing_only is False:
            assert report["step_time_reduction_fraction"] == 0.5
        else:
            assert "step_time_reduction_fraction" not in report
            assert set(report["conditions"]) == {"packed"}
    assert model.weight.item() == 2
    assert model.forward == original_forward
    assert model.chunk_gated_delta_rule is kernel
    assert torch._dynamo.config.recompile_limit == original_cache_limit
    assert torch._dynamo.config.fail_on_recompile_limit_hit == original_fail_on_limit


@pytest.mark.parametrize("value", [True, 129, 7, 64.0, "64"])
def test_compile_cache_limit_stays_bounded(value):
    with pytest.raises(ValueError):
        screen.validate_compile_cache_limit(value)


def receipt():
    return {
        "independent_loss": 0.9492,
        "packed_loss": 0.9671,
        "adapter_gradient_relative_l2": 0.1432,
        "cases": [
            {
                "repeat_max_abs": 0.0,
                "perturb_max_abs": 0.0,
                "cross_input_grad_max_abs": 0.0,
                "own_input_grad_max_abs": 1.0,
            }
        ],
    }


def test_learning_acceptance_preserves_strict_parity_failure():
    with pytest.raises(screen.PackingCanaryError):
        screen._accept_canary(receipt(), None)
    result = screen._accept_canary(receipt(), 0.15)
    assert result["passed"] is False
    assert result["accepted_for_learning_comparison"] is True
    assert result["learning_gradient_tolerance"] == 0.15


@pytest.mark.parametrize(
    "field,value",
    [
        ("adapter_gradient_relative_l2", 0.15001),
        ("adapter_gradient_relative_l2", float("nan")),
        ("packed_loss", 2.0),
        ("packed_loss", float("inf")),
        ("perturb_max_abs", 0.01),
        ("cross_input_grad_max_abs", 0.01),
        ("own_input_grad_max_abs", 0.0),
    ],
)
def test_learning_mode_still_rejects_leakage_and_unbounded_drift(field, value):
    row = receipt()
    if field in row:
        row[field] = value
    else:
        row["cases"][0][field] = value
    with pytest.raises(screen.PackingCanaryError):
        screen._accept_canary(row, 0.15)


@pytest.mark.parametrize("value", [True, 0.16, 0.01, float("nan"), "0.15"])
def test_learning_tolerance_rejects_unauthorized_ceiling(value):
    with pytest.raises(ValueError):
        screen.validate_learning_tolerance(value)
