"""Verify precision scope, closed loading contracts and matched job construction."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from experiments.b200_fp4_mlp_lora.run import candidate_job
from gleipnir.bf16_lora import (
    bf16_lora_metadata,
    fp4_mlp_lora_metadata,
    validate_fp4_mlp_lora_config,
)
from gleipnir.fouroversix_training import FrozenFourOverSixLinear
from gleipnir.monitoring_training_command import training_command
from gleipnir.packed_training import validate_packed_training_config
from tests.helpers.training import student_config


def mixed_student():
    student = student_config()
    student["quantization"].update(
        full_bf16_lora=False,
        fp4_mlp_lora=True,
        mlp_precision="fouroversix",
        fp4_backward_mode="dequantized_bf16",
        fp4_row_scaled_activations=True,
        fp4_fused_row_scaling=True,
    )
    return student


@pytest.mark.parametrize(
    "changes",
    [
        {"enabled": True},
        {"full_bf16_lora": True},
        {"mlp_precision": "bf16"},
        {"fp4_backward_mode": "fp4"},
        {"fp4_row_scaled_activations": False},
        {"fp4_activation_selector": "fp16"},
        {"fp4_mlp_lora": "true"},
    ],
)
def test_loading_rejects_mixed_or_unsupported_precision(changes):
    student = mixed_student()
    student["quantization"].update(changes)
    with pytest.raises(ValueError):
        validate_fp4_mlp_lora_config(student)


def test_historical_loading_and_strict_startup_remain_separate():
    assert not validate_fp4_mlp_lora_config(student_config())
    student = mixed_student()
    assert validate_fp4_mlp_lora_config(student)
    assert validate_packed_training_config(student)
    student["training"]["startup_validation_reference"] = "bf16_reference.json"
    with pytest.raises(ValueError, match="fresh startup"):
        validate_packed_training_config(student)


def native_model():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList()
    for _ in range(32):
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        for name in ["gate_proj", "up_proj", "down_proj"]:
            original = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
            original.requires_grad_(False)
            runtime = SimpleNamespace(
                row_scaled_activations=True,
                dequantized_weight=original.weight.detach().clone(),
                activation_selector="strict",
                activation_config=object(),
            )
            setattr(layer.mlp, name, FrozenFourOverSixLinear(original, runtime))
        model.layers.append(layer)
    model.lora_A = torch.nn.Parameter(torch.ones(2, 2))
    return model


def test_native_scope_cannot_be_mislabeled_as_full_bf16():
    model = native_model()
    metadata = fp4_mlp_lora_metadata(model)
    assert metadata["verified"] and len(metadata["native_mlp_modules"]) == 96
    assert metadata["master_dtype"] == "torch.float32"
    assert metadata["non_mlp_base_storage"] == "bf16"
    assert metadata["frozen_dtype"] != "torch.bfloat16"
    with pytest.raises(ValueError, match="native quantized"):
        bf16_lora_metadata(model)
    model.attention = model.layers[0].mlp.gate_proj
    with pytest.raises(ValueError, match="96 native"):
        fp4_mlp_lora_metadata(model)


def test_missing_or_mixed_native_mlps_and_master_dtype_fail():
    model = native_model()
    model.layers[0].mlp.gate_proj = torch.nn.Linear(
        2, 2, bias=False, dtype=torch.bfloat16
    )
    model.layers[0].mlp.gate_proj.requires_grad_(False)
    with pytest.raises(ValueError, match="96 native"):
        fp4_mlp_lora_metadata(model)
    model = native_model()
    model.lora_A.data = model.lora_A.data.to(torch.bfloat16)
    with pytest.raises(ValueError, match="FP32 master"):
        fp4_mlp_lora_metadata(model)


def test_job_keeps_workload_and_renders_no_kbit_loader():
    config = yaml.safe_load(
        Path("experiments/b200_fp4_mlp_lora/config.yaml").read_text()
    )
    source = dict(
        job_name="sdpa",
        seed=0,
        student_rows="rows",
        soft_targets="targets",
        max_length=29696,
        rank=128,
        lora_alpha=256,
        learning_rate=5e-5,
        num_train_epochs=-1,
        micro_batch_size=32,
        gradient_accumulation_steps=1,
        full_bf16_lora=True,
        sequence_packing=True,
        selection_sha256="frozen",
        startup_validation_reference="sdpa.json",
        packed_attention_backend="sdpa",
    )
    saved = deepcopy(source)
    job = candidate_job(config, {"job": source})
    assert source == saved
    assert job["selection_sha256"] == "frozen" and job["micro_batch_size"] == 32
    assert job["sequence_packing"] and "startup_validation_reference" not in job
    assert "packing_learning_gradient_tolerance" not in job
    command = training_command(job)
    assert "student.quantization.enabled=false" in command
    assert "++student.quantization.fp4_mlp_lora=true" in command
    assert "++student.quantization.full_bf16_lora=false" in command
    assert "++student.quantization.mlp_precision=fouroversix" in command
    assert "++student.quantization.fp4_backward_mode=dequantized_bf16" in command
