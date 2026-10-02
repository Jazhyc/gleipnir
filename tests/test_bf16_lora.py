"""Verify that the ordinary LoRA experiment really removes base quantization."""

from pathlib import Path

import pytest
import torch
import yaml

from experiments.fp4_stability.run import campaign_stages, validate_config
from gleipnir.bf16_lora import bf16_lora_metadata, configure_bf16_reductions


def test_bf16_reduction_policy_controls_split_k_separately():
    before = configure_bf16_reductions()
    before_library = torch.backends.cuda.preferred_blas_library()
    try:
        assert configure_bf16_reductions(
            allow_reduced_precision=False, allow_split_k=False
        ) == {
            "allow_reduced_precision": False,
            "allow_split_k": False,
            "blas_library": "cublaslt",
        }
        with pytest.raises(ValueError, match="disabling split-K"):
            configure_bf16_reductions(allow_reduced_precision=True)
        assert configure_bf16_reductions()["allow_split_k"] is False
        assert (
            configure_bf16_reductions(allow_reduced_precision=True, allow_split_k=True)[
                "allow_split_k"
            ]
            is True
        )
        with pytest.raises(ValueError, match="booleans"):
            configure_bf16_reductions(allow_split_k="false")
    finally:
        configure_bf16_reductions(
            allow_reduced_precision=before["allow_reduced_precision"],
            allow_split_k=before["allow_split_k"],
        )
        torch.backends.cuda.preferred_blas_library(before_library)


def make_model():
    model = torch.nn.ModuleDict(
        {
            "base": torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16),
            "lora_A": torch.nn.Linear(2, 2, bias=False),
        }
    )
    model["base"].requires_grad_(False)
    return model


def test_full_base_precision_and_master_dtype_are_verified():
    model = make_model()
    assert bf16_lora_metadata(model) == {
        "verified": True,
        "frozen_dtype": "torch.bfloat16",
        "frozen_elements": 4,
        "master_dtype": "torch.float32",
        "trainable_elements": 4,
        "quantized_modules": [],
    }
    model["base"].float()
    with pytest.raises(ValueError, match="BF16 frozen bases"):
        bf16_lora_metadata(model)
    model = make_model()
    model["lora_A"].bfloat16()
    with pytest.raises(ValueError, match="FP32 master"):
        bf16_lora_metadata(model)


def test_quantized_module_and_unfrozen_base_are_rejected():
    model = make_model()
    model.add_module(
        "quantized_attention", type("Linear4bit", (torch.nn.Module,), {})()
    )
    with pytest.raises(ValueError, match="no quantized"):
        bf16_lora_metadata(model)
    model = make_model()
    model["base"].requires_grad_(True)
    with pytest.raises(ValueError, match="FP32 master"):
        bf16_lora_metadata(model)


def test_full_bf16_campaign_has_exactly_ten_calls_and_keeps_batch_policy():
    path = Path(__file__).parents[1] / "experiments/fp4_stability"
    full = yaml.safe_load((path / "bf16_flashqla_ten_step_comparison.yaml").read_text())
    control = yaml.safe_load(
        (path / "nf4_flashqla_ten_step_comparison.yaml").read_text()
    )
    validate_config(full)
    stages = campaign_stages(full, {"selection": "320"}, {"selection": "longest"})
    assert len(stages) == 1 and stages[0]["precision"] == "bf16"
    assert stages[0]["steps"] == 10
    differences = {key for key in full if full[key] != control.get(key)}
    assert differences == {"output", "logs", "conditions", "full_bf16_lora"}
    for overrides in [
        {"full_bf16_lora": False},
        {"conditions": ["nf4"]},
        {"steps": 1},
        {"ten_step_learning_comparison": False},
    ]:
        with pytest.raises(ValueError, match="ten-step comparison"):
            validate_config({**full, **overrides})
