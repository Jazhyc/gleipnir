"""Check that the selected B200 recipe reaches training and keeps truthful gates."""

from pathlib import Path

import pytest
import torch

from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)
from gleipnir.flashqla_training import (
    FLASHQLA_TARGET,
    flashqla_environment,
    install_with_model_canary,
    selected_recipe_canary_accepted,
)
from gleipnir.monitoring_systems_screen import load_config, make_jobs, resolve_paths


def test_selected_profile_composes_and_forwards_the_backend():
    source = Path("experiments/b200_adaptive_microbatching/config.yaml")
    original = load_config(source)
    selected = load_config(
        source, overrides=["systems_screen@_global_=qwen35_4b_b200_default"]
    )
    reference = load_config(
        source, overrides=["systems_screen@_global_=qwen35_4b_b200_adaptive"]
    )
    assert selected["data"] == original["data"]
    assert selected["recipe"] == {
        **reference["recipe"],
        "gated_delta_backend": "flashqla",
        "gated_delta_parity_policy": "selected_finite",
    }
    assert "gated_delta_backend" not in original["recipe"]
    assert "gated_delta_backend" not in reference["recipe"]
    assert (
        selected["metadata_expectations"]["gated_delta_backend.replaced_layers"] == 24
    )
    for job in make_jobs(selected, resolve_paths(selected), "fixed"):
        command = training_command(job)
        assert "++student.training.gated_delta_backend=flashqla" in command
        assert "++student.training.gated_delta_parity_policy=selected_finite" in command
    env = {"PYTHONPATH": "pinned-fla:project", "CUDA_VISIBLE_DEVICES": "0"}
    updated = flashqla_environment(env)
    assert (
        updated["PYTHONPATH"] == str(FLASHQLA_TARGET.resolve()) + ":pinned-fla:project"
    )
    assert (
        updated["CUDA_VISIBLE_DEVICES"] == "0"
        and env["PYTHONPATH"] == "pinned-fla:project"
    )


@pytest.mark.parametrize("finite", [False, True])
def test_finite_acceptance_does_not_relabel_strict_failure(finite):
    receipt = {"passed": False, "finite": finite}
    assert not selected_recipe_canary_accepted(receipt, selected=False)
    assert selected_recipe_canary_accepted(receipt, selected=True) is finite
    assert receipt["passed"] is False
    assert receipt["accepted_for_selected_recipe"] is finite
    assert not selected_recipe_canary_accepted({"passed": False}, selected=True)


@pytest.mark.parametrize("factor", [2.0, 0.0, float("inf")])
def test_selected_model_canary_accepts_finite_nonzero_gradients_and_restores_failures(
    monkeypatch, factor
):
    import gleipnir.flashqla_training as backend

    def reference(q, *args, **kwargs):
        return q, None

    reference.__module__ = "fla.ops.gated_delta_rule"
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.tensor(1.0))
    model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(24)])
    for layer in model.layers:
        layer.chunk_gated_delta_rule = reference

    def forward(value):
        q = value * model.weight
        return torch.stack(
            [layer.chunk_gated_delta_rule(q, q, q, q, q)[0] for layer in model.layers]
        ).mean()

    monkeypatch.setattr(
        backend, "load_flashqla", lambda: (lambda q, **kw: (q * factor, None), {})
    )
    kwargs = dict(
        auto_cp=False,
        bf16_boundary=True,
        boundary_policy="bf16_fp32_gates_norm",
        selected_recipe=True,
    )
    if factor == 2:
        receipt = install_with_model_canary(
            model, [torch.tensor(1.0)], forward, **kwargs
        )
        assert not receipt["passed"] and receipt["accepted_for_selected_recipe"]
        assert receipt["replaced_layers"] == 24
        assert all(
            layer.chunk_gated_delta_rule is not reference for layer in model.layers
        )
    else:
        with pytest.raises(FloatingPointError):
            install_with_model_canary(model, [torch.tensor(1.0)], forward, **kwargs)
        assert all(layer.chunk_gated_delta_rule is reference for layer in model.layers)
    assert model.weight.grad is None and model.weight.item() == 1 and model.training
