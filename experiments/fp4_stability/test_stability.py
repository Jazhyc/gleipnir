"""Check the conservative backward contract independently of CUDA packing."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from experiments.fp4_stability.run import campaign_stages, validate_config
from gleipnir.fouroversix_training import (
    FrozenFourOverSixLinear,
    FrozenFp4Runtime,
    normalize_activation_rows,
)


def test_configs_are_bounded_and_invalid_campaign_fails_before_loading():
    root = Path(__file__).parent
    for name in [
        "config.yaml",
        "row_dequantized_diagnostic.yaml",
        "precision_cast_diagnostic.yaml",
    ]:
        config = yaml.safe_load((root / name).read_text())
        validate_config(config)
        assert config["diagnostics_only"]
        with pytest.raises(ValueError, match="global preflight"):
            validate_config({**config, "steps": 10})
        with pytest.raises(ValueError, match="ten matched"):
            validate_config({**config, "steps": 1000})
    training = yaml.safe_load((root / "row_dequantized_training.yaml").read_text())
    validate_config(training)
    assert not training["diagnostics_only"] and training["steps"] == 10
    matched_diagnostic = yaml.safe_load(
        (root / "row_precision_cast_diagnostic.yaml").read_text()
    )
    validate_config(matched_diagnostic)
    assert matched_diagnostic["diagnostics_only"]
    for filename in [
        "row_aot_diagnostic.yaml",
        "row_operand_diagnostic.yaml",
        "row_eager_norm_diagnostic.yaml",
    ]:
        validate_config(yaml.safe_load((root / filename).read_text()))
    matched_nf4 = yaml.safe_load((root / "row_aot_nf4_matched.yaml").read_text())
    validate_config(matched_nf4)
    with pytest.raises(ValueError, match="expected master hash"):
        validate_config({**matched_nf4, "expected_initial_master_sha256": None})


@pytest.mark.parametrize("observe", [False, True])
def test_dequantized_backward_uses_forward_weight_not_master_or_transpose(observe):
    torch.manual_seed(19)
    original = torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)
    original.requires_grad_(False)
    decoded = original.weight.detach().clone() + 0.125
    calls = []

    def native_matmul(inputs, weight, *, input_config):
        calls.append("native_forward")
        return inputs @ weight.T

    runtime = FrozenFp4Runtime(
        decoded,
        torch.zeros(16, 32, dtype=torch.bfloat16),
        None,
        None,
        native_matmul,
        dequantized_weight=decoded,
    )
    layer = FrozenFourOverSixLinear(original, runtime)
    observations = []
    if observe:
        runtime.observer = lambda *args: observations.append(args[0].detach().clone())
    inputs = torch.randn(2, 7, 16, dtype=torch.bfloat16, requires_grad=True)
    gradient = torch.randn(2, 7, 32, dtype=torch.bfloat16)
    layer(inputs).backward(gradient)
    torch.testing.assert_close(inputs.grad, gradient @ decoded)
    assert not torch.equal(inputs.grad, gradient @ original.weight)
    assert original.weight.grad is None
    assert calls == ["native_forward"]
    assert runtime.forward_calls == runtime.backward_calls == 1
    assert len(observations) == int(observe)
    if observe:
        torch.testing.assert_close(observations[0], inputs.detach())


def test_operand_metrics_record_value_drift_and_reject_different_shapes():
    from gleipnir.fp4_compiler_diagnostic import tensor_difference

    eager = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    compiled = eager.float() + torch.tensor([0.0, 0.0, 0.0, 1.0])
    metrics = tensor_difference(eager, compiled)
    assert metrics["unequal_fraction"] == 0.25
    assert metrics["maximum_absolute_difference"] == 1.0
    assert metrics["relative_l2"] == pytest.approx(1 / 30**0.5)
    assert metrics["eager_dtype"] != metrics["compiled_dtype"]
    with pytest.raises(ValueError, match="shapes differ"):
        tensor_difference(eager, compiled.reshape(2, 2))


@pytest.mark.parametrize("diagnostics_only", [False, True])
def test_initialization_mismatch_fails_before_model_or_optimizer_calls(
    tmp_path, diagnostics_only
):
    from gleipnir.adaptive_microbatching import MicrobatchPolicy
    from gleipnir.precision_training_screen import run_precision_training_screen

    model = torch.nn.Linear(2, 2, bias=False)
    before = model.weight.detach().clone()

    def forbidden(*args):
        pytest.fail("initialization mismatch must fail before model or optimizer work")

    with pytest.raises(ValueError, match="initial adapter hash mismatch"):
        run_precision_training_screen(
            model=model,
            features=[{"direct_input_ids": [1]} for _ in range(32)],
            collator=forbidden,
            loss_forward=forbidden,
            optimizer_factory=forbidden,
            scheduler_factory=forbidden,
            install_compile=forbidden,
            output=tmp_path,
            seed=0,
            policy=MicrobatchPolicy(max_padded_tokens=16384, max_micro_batch_size=8),
            steps=1,
            max_grad_norm=1.0,
            metadata={},
            diagnostics_only=diagnostics_only,
            expected_initial_master_sha256="0" * 64,
        )
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
    assert model.weight.grad is None
    assert not (tmp_path / "screen.json").exists()


def test_eager_norm_boundary_preserves_parameters_and_leaves_gated_norm(monkeypatch):
    from gleipnir.fouroversix_training import install_eager_rmsnorm_interfaces

    class Qwen3_5RMSNorm(torch.nn.Linear):
        pass

    class Qwen3_5RMSNormGated(torch.nn.Linear):
        pass

    model = torch.nn.ModuleDict(
        {
            "input_norm": Qwen3_5RMSNorm(2, 2),
            "gated_norm": Qwen3_5RMSNormGated(2, 2),
        }
    )
    keys = list(model.state_dict())
    parameters = list(model.parameters())
    disabled = []
    monkeypatch.setattr(torch.compiler, "disable", lambda fn: disabled.append(fn) or fn)
    assert install_eager_rmsnorm_interfaces(model) == ["input_norm"]
    assert len(disabled) == 1
    assert list(model.state_dict()) == keys
    assert all(p is q for p, q in zip(parameters, model.parameters(), strict=True))
    with pytest.raises(ValueError, match="no Qwen3.5"):
        install_eager_rmsnorm_interfaces(torch.nn.Linear(2, 2))


def test_ten_update_campaign_requires_global_preflight_before_each_condition():
    source, longest = {"selection": "320"}, {"selection": "global32"}
    stages = campaign_stages(
        {"steps": 10, "conditions": ["fouroversix", "bf16"]}, source, longest
    )
    assert [(s["name"], s["steps"], s["source"]) for s in stages] == [
        ("fouroversix-global-preflight", 1, longest),
        ("fouroversix", 10, source),
        ("bf16-global-preflight", 1, longest),
        ("bf16", 10, source),
    ]
    assert campaign_stages(
        {"steps": 1, "conditions": ["fouroversix"]}, longest, longest
    ) == [dict(name="fouroversix", precision="fouroversix", source=longest, steps=1)]
    assert campaign_stages(
        {"steps": 10, "conditions": ["fouroversix"], "diagnostics_only": True},
        source,
        longest,
    ) == [dict(name="fouroversix", precision="fouroversix", source=source, steps=10)]


def test_timing_summary_weights_tokens_and_rejects_unmatched_replays():
    from copy import deepcopy

    from gleipnir.precision_training_screen import timing_summary

    passes = [
        {
            "new_dynamo_graphs": 0,
            "steps": [
                {
                    "seconds": duration,
                    "actual_tokens": 100 * (i + 1),
                    "dataset_indices": [i],
                    "padded_tokens": 100 * (i + 1),
                    "physical_sizes": [1],
                    "learning_rate": 0.0 if i == 0 else 5e-5,
                }
                for i in range(10)
            ],
        }
        for duration in [1.0, 2.0, 3.0]
    ]
    summary = timing_summary(passes)
    assert summary["measured_steps"] == 30
    assert summary["mean_step_seconds"] == summary["median_step_seconds"] == 2.0
    assert summary["actual_tokens_per_second"] == 16500 / 60
    assert summary["per_batch_mean_seconds"] == [2.0] * 10
    bad = deepcopy(passes)
    bad[1]["steps"][0]["learning_rate"] = 5e-5
    with pytest.raises(ValueError, match="learning_rate"):
        timing_summary(bad)
    with pytest.raises(ValueError, match="complete ten-step"):
        timing_summary([{**passes[0], "steps": passes[0]["steps"][:-1]}])


@pytest.mark.parametrize("ten_step_comparison", [False, True])
@pytest.mark.parametrize("profile_batch", [None, 5])
@pytest.mark.parametrize("gradient_validation", ["per_tensor", "clip_norm"])
def test_timing_replays_restore_adapters_optimizer_and_schedule(
    monkeypatch, tmp_path, profile_batch, gradient_validation, ten_step_comparison
):
    if ten_step_comparison and profile_batch is not None:
        pytest.skip("bounded comparison excludes profiling")
    from gleipnir import precision_training_screen as screen
    from gleipnir.adaptive_microbatching import MicrobatchPolicy

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "cpu mock")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=0),
    )
    for name in ["synchronize", "reset_peak_memory_stats", "manual_seed_all"]:
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    for name in ["max_memory_allocated", "max_memory_reserved"]:
        monkeypatch.setattr(torch.cuda, name, lambda *args: 0)
    monkeypatch.setattr(screen.importlib.metadata, "version", lambda name: "mock")
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizers = []
    profiled = []

    def profile_action(action, output):
        profiled.append(action())
        return {"scope": "mock backward only"}

    monkeypatch.setattr("gleipnir.fp4_performance.profile_backward", profile_action)

    def optimizer_factory():
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        optimizers.append(optimizer)
        return optimizer

    report = screen.run_precision_training_screen(
        model=model,
        features=[{"direct_input_ids": [1, 2]} for _ in range(320)],
        collator=lambda items: items,
        loss_forward=lambda batch: model(torch.ones(1, 1)).square().mean(),
        optimizer_factory=optimizer_factory,
        scheduler_factory=lambda optimizer, steps: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 0.0 if step == 0 else (10 - step) / 9
        ),
        install_compile=lambda: [],
        output=tmp_path,
        seed=0,
        policy=MicrobatchPolicy(max_padded_tokens=16384, max_micro_batch_size=8),
        steps=10,
        max_grad_norm=1.0,
        metadata={"mlp": {"precision": "nf4" if ten_step_comparison else "bf16"}},
        timing_repeats=0 if ten_step_comparison else 3,
        ten_step_learning_comparison=ten_step_comparison,
        profile_batch=profile_batch,
        gradient_validation=gradient_validation,
    )
    assert report["status"] == "complete"
    if ten_step_comparison:
        assert len(optimizers) == 1
        assert report["compile_warmup_master_unchanged"]
        assert len(report["compile_warmup"]) == 10
        assert all(row["optimizer_updates"] == 0 for row in report["compile_warmup"])
        assert int(optimizers[0].state[model.weight]["step"]) == 10
        assert report["timing_summary"]["measured_steps"] == 10
        assert all("common_probe" in row for row in report["steps"])
        return
    assert len(optimizers) == 4
    assert len(profiled) == int(profile_batch is not None)
    assert len({id(item.state) for item in optimizers}) == 4
    assert report["timing_passes"][0]["kind"] == "warmup"
    assert report["timing_summary"]["measured_steps"] == 30
    assert len({p["final_master_sha256"] for p in report["timing_passes"]}) == 1
    for replay in report["timing_passes"]:
        assert replay["initial_master_sha256"] == report["initial_master_sha256"]
        assert replay["steps"][0]["learning_rate"] == 0.0
        assert replay["steps"][1]["learning_rate"] == 5e-5
        assert [step["mean_loss"] for step in replay["steps"]] == [
            step["mean_loss"] for step in report["timing_passes"][0]["steps"]
        ]


@pytest.mark.parametrize("gradient_validation", ["per_tensor", "clip_norm"])
def test_nonfinite_gradient_fails_before_optimizer(
    monkeypatch, tmp_path, gradient_validation
):
    from gleipnir import precision_training_screen as screen
    from gleipnir.adaptive_microbatching import MicrobatchPolicy

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "cpu mock")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=0),
    )
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(screen.importlib.metadata, "version", lambda name: "mock")
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))

    def forbidden(*args):
        pytest.fail("nonfinite gradients must fail before optimizer construction")

    with pytest.raises(
        (FloatingPointError, RuntimeError), match="nonfinite|non-finite"
    ):
        screen.run_precision_training_screen(
            model=model,
            features=[{"direct_input_ids": [1]} for _ in range(32)],
            collator=lambda items: items,
            loss_forward=lambda batch: model(torch.ones(1, 1)).square().mean(),
            optimizer_factory=forbidden,
            scheduler_factory=forbidden,
            install_compile=lambda: [],
            output=tmp_path,
            seed=0,
            policy=MicrobatchPolicy(max_padded_tokens=16384, max_micro_batch_size=8),
            steps=1,
            max_grad_norm=1.0,
            metadata={"mlp": {"precision": "bf16"}},
            gradient_validation=gradient_validation,
        )


def test_fused_scaling_wrapper_preserves_forward_and_decoded_backward(monkeypatch):
    import sys

    fake = SimpleNamespace(
        normalize_rows=normalize_activation_rows,
        rescale_rows=lambda outputs, scales: (outputs.float() * scales).to(
            torch.bfloat16
        ),
    )
    monkeypatch.setitem(sys.modules, "gleipnir.fp4_row_kernels", fake)
    original = torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)
    original.requires_grad_(False)
    layers = []
    for fused in [False, True]:
        runtime = FrozenFp4Runtime(
            original.weight,
            None,
            None,
            None,
            lambda inputs, weight, **kwargs: inputs @ weight.T,
            dequantized_weight=original.weight,
            row_scaled_activations=True,
            fused_row_scaling=fused,
        )
        layers.append(FrozenFourOverSixLinear(original, runtime))
    values = torch.randn(2, 7, 16, dtype=torch.bfloat16)
    values[0, 0] = 0
    outputs, gradients = [], []
    for layer in layers:
        inputs = values.clone().requires_grad_()
        output = layer(inputs)
        output.float().sum().backward()
        outputs.append(output.detach())
        gradients.append(inputs.grad)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)


def test_packed_wrapper_passes_prepacked_input_and_preserves_backward(monkeypatch):
    import sys

    calls = []

    def pack(inputs, fixed_amax, *, implementation):
        assert implementation == "tiled"
        normalized, scales = normalize_activation_rows(inputs)
        payload = SimpleNamespace(decoded=normalized)
        calls.append(payload)
        return payload, scales

    monkeypatch.setitem(
        sys.modules,
        "gleipnir.fp4_quantization_kernels",
        SimpleNamespace(quantize_activation_rows=pack),
    )
    monkeypatch.setitem(
        sys.modules,
        "gleipnir.fp4_row_kernels",
        SimpleNamespace(
            rescale_rows=lambda output, scales: (output.float() * scales).to(
                torch.bfloat16
            )
        ),
    )
    original = torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)
    original.requires_grad_(False)

    def matmul(payload, weight, *, input_config):
        assert payload is calls[-1]
        return payload.decoded @ weight.T

    runtime = FrozenFp4Runtime(
        original.weight,
        None,
        SimpleNamespace(kwargs={"x_amax": torch.ones(1)}),
        None,
        matmul,
        dequantized_weight=original.weight,
        row_scaled_activations=True,
        fused_row_scaling=True,
        fused_activation_packing=True,
    )
    layer = FrozenFourOverSixLinear(original, runtime)
    inputs = torch.randn(2, 7, 16, dtype=torch.bfloat16, requires_grad=True)
    normalized, scales = normalize_activation_rows(inputs.detach().reshape(-1, 16))
    expected = ((normalized @ original.weight.T).float() * scales).to(torch.bfloat16)
    output = layer(inputs)
    torch.testing.assert_close(output.reshape(-1, 32), expected, rtol=0, atol=0)
    gradient = torch.randn_like(output)
    output.backward(gradient)
    torch.testing.assert_close(inputs.grad, gradient @ original.weight, rtol=0, atol=0)
    assert len(calls) == 1 and original.weight.grad is None
    runtime.observer = lambda *args: None
    with pytest.raises(ValueError, match="observation"):
        layer(inputs)


def test_packing_campaign_rejects_inconsistent_precision_and_observation():
    config = yaml.safe_load(
        (Path(__file__).parent / "row_inductor_packed_timing.yaml").read_text()
    )
    validate_config(config)
    for overrides in [
        {"row_scaled_activations": False},
        {"fused_row_scaling": False},
        {"backward_mode": "fp4"},
        {"conditions": ["bf16"]},
        {"capture_native_operands": True},
    ]:
        with pytest.raises(ValueError, match="per-token|fused packing"):
            validate_config({**config, **overrides})


def test_timing_config_is_fp4_only_and_bounded():
    config = yaml.safe_load((Path(__file__).parent / "row_aot_timing.yaml").read_text())
    validate_config(config)
    assert config["conditions"] == ["fouroversix"]
    assert config["timing_repeats"] == 3
    for overrides in [
        {"timing_repeats": 100},
        {"diagnostics_only": True},
        {"steps": 1},
    ]:
        with pytest.raises(ValueError, match="timing benchmark|global preflight"):
            validate_config({**config, **overrides})
    profile = yaml.safe_load(
        (Path(__file__).parent / "row_aot_profile.yaml").read_text()
    )
    validate_config(profile)
    for overrides in [{"profile_batch": 11}, {"steps": 1}]:
        with pytest.raises(ValueError, match="profiling requires"):
            validate_config({**profile, **overrides})
    fused = yaml.safe_load(
        (Path(__file__).parent / "row_aot_fused_timing.yaml").read_text()
    )
    validate_config(fused)
    with pytest.raises(ValueError, match="per-token"):
        validate_config({**fused, "row_scaled_activations": False})
    with pytest.raises(ValueError, match="validation mode"):
        validate_config({**fused, "gradient_validation": "disabled"})
    validate_config(
        yaml.safe_load(
            (
                Path(__file__).parent / "row_inductor_boundaries_diagnostic.yaml"
            ).read_text()
        )
    )


def test_eager_mlp_activation_boundary_preserves_parameters(monkeypatch):
    from gleipnir.fouroversix_training import install_eager_mlp_activation_interfaces

    model = torch.nn.ModuleDict(
        {
            "layer": torch.nn.ModuleDict(
                {
                    "mlp": torch.nn.ModuleDict(
                        {
                            "act_fn": torch.nn.SiLU(),
                            "projection": torch.nn.Linear(2, 2),
                        }
                    )
                }
            ),
            "other": torch.nn.SiLU(),
        }
    )
    parameters = list(model.parameters())
    disabled = []
    monkeypatch.setattr(torch.compiler, "disable", lambda fn: disabled.append(fn) or fn)
    assert install_eager_mlp_activation_interfaces(model) == ["layer.mlp.act_fn"]
    assert len(disabled) == 1
    assert all(p is q for p, q in zip(parameters, model.parameters(), strict=True))


def test_row_scaling_is_independent_of_other_tokens_and_handles_zero():
    inputs = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, -4.0, 0.0]], dtype=torch.bfloat16
    )
    normalized, scales = normalize_activation_rows(inputs)
    extended, extended_scales = normalize_activation_rows(
        torch.cat([inputs, torch.full((1, 4), 1e8, dtype=torch.bfloat16)])
    )
    torch.testing.assert_close(normalized, extended[:2], rtol=0, atol=0)
    torch.testing.assert_close(scales, extended_scales[:2], rtol=0, atol=0)
    torch.testing.assert_close(normalized.float() * scales, inputs.float())
    assert torch.isfinite(normalized).all()
    assert torch.equal(scales, torch.tensor([[1.0], [4.0]]))


@pytest.mark.parametrize("diagnostics_only", [False, True])
def test_failed_gate_never_updates_adapters(
    monkeypatch, tmp_path: Path, diagnostics_only
):
    from gleipnir import precision_training_screen as screen
    from gleipnir.adaptive_microbatching import MicrobatchPolicy

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "cpu mock")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=0),
    )
    monkeypatch.setattr(screen.importlib.metadata, "version", lambda name: "mock")
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(
        [torch.nn.Linear(1, 1, bias=False) for _ in range(32)]
    )
    model.model = SimpleNamespace(layers=model.layers)
    with torch.no_grad():
        for layer in model.layers:
            layer.weight.fill_(1.0)

    def loss_forward(batch):
        value = torch.ones(1, 1)
        for layer in model.layers:
            value = layer(value)
        return value.sum()

    def install_compile():
        original = model.layers[0].forward
        model.layers[0].forward = lambda x: original(x) * 2
        return list(range(32))

    def forbidden(*args):
        pytest.fail("diagnostics must not create an optimizer or scheduler")

    arguments = dict(
        model=model,
        features=[{"direct_input_ids": list(range(16))} for _ in range(32)],
        collator=lambda items: items,
        loss_forward=loss_forward,
        optimizer_factory=forbidden,
        scheduler_factory=forbidden,
        install_compile=install_compile,
        output=tmp_path,
        seed=0,
        policy=MicrobatchPolicy(max_padded_tokens=16384, max_micro_batch_size=8),
        steps=1,
        max_grad_norm=1.0,
        metadata={},
        diagnostics_only=diagnostics_only,
    )
    if diagnostics_only:
        report = screen.run_precision_training_screen(**arguments)
        assert report["status"] == "diagnosed"
        assert report["prefix_compile_losses"][0]["loss"] == 1.0
        assert report["prefix_compile_losses"][1]["loss"] == 2.0
        assert report["master_unchanged"]
    else:
        with pytest.raises(ValueError, match="compilation loss canary failed"):
            screen.run_precision_training_screen(**arguments)
        report = json.loads((tmp_path / "screen.json").read_text())
        assert report["status"] == "failed"
        assert all(layer.weight.item() == 1.0 for layer in model.layers)
        assert all(layer.weight.grad is None for layer in model.layers)
    assert report["compile_canary"]["passed"] is False
    assert report["compile_canary"]["eager_repeat_loss"] == 1.0
    assert report["compile_canary"]["compiled_repeat_loss"] == 2.0
    assert report["steps"] == []


def test_explicit_ten_step_comparison_has_no_extra_optimizer_stage():
    root = Path(__file__).parent
    for name in ["nf4_flashqla_ten_step_comparison", "nf4_fla_ten_step_comparison"]:
        config = yaml.safe_load((root / f"{name}.yaml").read_text())
        validate_config(config)
        stages = campaign_stages(config, {"selection": "320"}, {"selection": "longest"})
        assert len(stages) == 1 and stages[0]["steps"] == 10
        for overrides in [
            {"steps": 1},
            {"timing_repeats": 3},
            {"fp32_lm_head": True},
            {"conditions": ["bf16"]},
            {"flashqla_layer_indices": [30]},
            {"diagnostics_only": True},
        ]:
            with pytest.raises(ValueError, match="ten-step comparison"):
                validate_config({**config, **overrides})
