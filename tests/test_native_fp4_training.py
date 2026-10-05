"""Exercise the ordinary FP4 recipe, receipt identity and retained runtime contract."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gleipnir import native_fp4_training as native
from gleipnir.monitoring_training_command import training_command
from gleipnir.packed_training import validate_packed_training_config
from gleipnir.validated_startup import validation_reference

ROOT = Path(__file__).resolve().parents[1]


def test_targeted_equivalence_binds_sources_and_rejects_drift(monkeypatch):
    native.validate_kernel_sources()
    assert native.KERNEL_SHA256 != native.HISTORICAL_KERNEL_SHA256
    proof = native.RUNTIME_SHAPE_VALIDATION
    assert proof["performed_this_run"] is False
    assert sum(r["cases"] for r in proof["receipts"].values()) == 44
    monkeypatch.setattr(
        native, "KERNEL_SHA256", {**native.KERNEL_SHA256, "cudnn_fp4_mlp.py": "drift"}
    )
    with pytest.raises(ValueError, match="kernel source changed"):
        native.validate_kernel_sources()


def profile(name="qwen35_4b_b200_default"):
    with initialize_config_dir(
        config_dir=str(ROOT / "src/gleipnir/configs/systems_screen"),
        version_base=None,
    ):
        return OmegaConf.to_container(compose(config_name=name), resolve=True)


def student_config(max_steps=512):
    job = {
        **profile()["recipe"],
        "job_name": "native-fp4-test",
        "output_dir": "results/native-fp4-test",
        "student_rows": "data/test.jsonl",
        "soft_targets": "data/soft.jsonl",
        "causal_adapter_dir": "results/native-fp4-test/causal_adapter",
        "max_steps": max_steps,
    }
    command = training_command(job)
    assert command[1] == "experiments/deception_distillation/train_student_sft.py"
    assert "++student.training.native_fp4_mlp=true" in command
    assert "++student.training.native_fp4_mlp_parity_policy=selected_finite" in command
    with initialize_config_dir(
        config_dir=str(ROOT / "experiments/tool_trajectory_monitoring"),
        version_base=None,
    ):
        cfg = compose(config_name="distillation_config", overrides=command[6:])
        return OmegaConf.to_container(cfg.student, resolve=True)


@pytest.mark.parametrize("max_steps", [-1, 512])
def test_default_reaches_the_ordinary_trainer_without_the_timing_limit(max_steps):
    assert profile() == profile("qwen35_4b_b200_fp4_mlp")
    student = student_config(max_steps)
    assert validate_packed_training_config(student)
    assert student["training"]["max_steps"] == max_steps
    assert "packing_timing_authority" not in student["training"]
    bf16 = profile("qwen35_4b_b200_bf16_fa4")
    assert not bf16["recipe"].get("native_fp4_mlp", False)
    assert bf16["recipe"]["packing_learning_gradient_tolerance"] == 0.10


@pytest.mark.parametrize(
    "field,value",
    [
        ("training.native_fp4_mlp_parity_policy", "strict"),
        ("training.sequence_packing", False),
        ("training.startup_validation_reference_sha256", "different"),
        ("training.startup_validation_reference", None),
        ("training.selective_torch_compile_mode", "reduce-overhead"),
        ("training.selective_torch_compile_dynamic", False),
        ("training.gradient_checkpointing", True),
        ("training.packed_attention_backend", "sdpa"),
        ("training.packing_learning_gradient_tolerance", 0.10),
        ("training.packing_timing_authority", "benchmark only"),
        ("training.per_device_train_batch_size", 16),
        ("training.adaptive_microbatching.max_padded_tokens", 8192),
        ("training.adaptive_microbatching.max_micro_batch_size", 4),
        ("quantization.enabled", True),
        ("model", "Qwen/Qwen3.5-9B"),
        ("model_revision", "different"),
        ("lora.r", 256),
        ("lora.dropout", 0.1),
        ("max_length", 30000),
    ],
)
def test_changed_recipe_stops_before_model_loading(field, value):
    student = student_config()
    target = student
    components = field.split(".")
    for component in components[:-1]:
        target = target[component]
    target[components[-1]] = value
    with pytest.raises(ValueError, match="native FP4"):
        validate_packed_training_config(student)


def packing_canary():
    return {
        "passed": False,
        "accepted_for_learning_comparison": False,
        "accepted_for_timing_comparison": True,
        "independent_loss": 0.9,
        "packed_loss": 1.1,
        "adapter_gradient_relative_l2": 0.65,
        "cases": [
            {
                "repeat_max_abs": 0.0,
                "perturb_max_abs": 0.0,
                "cross_input_grad_max_abs": 0.0,
                "own_input_grad_max_abs": 1.0,
            }
        ],
    }


def test_selected_acceptance_preserves_failed_parity_and_rejects_bad_isolation():
    receipt = packing_canary()
    original = deepcopy(receipt)
    native.accept_selected_canary(receipt)
    assert receipt == original and receipt["passed"] is False
    for change in ("leak", "nonfinite", "missing_own_gradient"):
        invalid = deepcopy(receipt)
        if change == "leak":
            invalid["cases"][0]["cross_input_grad_max_abs"] = 0.1
        elif change == "nonfinite":
            invalid["adapter_gradient_relative_l2"] = float("nan")
        else:
            invalid["cases"][0]["own_input_grad_max_abs"] = 0
        with pytest.raises(ValueError):
            native.accept_selected_canary(invalid)


def test_native_receipt_is_bound_and_cannot_authorize_bf16(tmp_path, monkeypatch):
    metadata = {
        "model": "Qwen/Qwen3.5-4B",
        "model_revision": "revision",
        "training_state": {"global_step": 20},
        "gradient_checkpointing": False,
        "quantization": {
            "enabled": False,
            "full_bf16_lora": {
                "verified": True,
                "native_fp4_mlp": {"modules": ["mlp"]},
            },
        },
        "gated_delta_backend": {
            "backend": "flashqla",
            "replaced_layers": 24,
            "boundary_policy": "bf16_fp32_gates_norm",
            "auto_cp": False,
            "finite": True,
            "passed": False,
        },
        "sequence_packing": {
            "attention_backend": "flash_attention_4",
            "attention_version": "4.0.0b33",
            "learning_gradient_tolerance": 0.05,
            "max_packed_tokens": 16384,
            "eager_canary": packing_canary(),
            "compiled_canary": packing_canary(),
            "preflight": {"passed": True},
        },
    }
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="MLP precision"):
        validation_reference(path)
    with pytest.raises(ValueError, match="checksum-bound"):
        validation_reference(path, native_fp4_mlp=True)
    with pytest.raises(ValueError, match="validated warmed receipt"):
        validation_reference(path, native_fp4_mlp=True, expected_sha256=digest)
    checked = []
    monkeypatch.setattr(
        native, "validate_native_fp4_reference", lambda m, d: checked.append(d)
    )
    result = validation_reference(path, native_fp4_mlp=True, expected_sha256=digest)
    assert checked == [digest]
    assert result["policy"] == "reuse_selected_finite_native_fp4_recipe"
    assert result["performed_this_run"] is False
    assert result["reference_packing"]["eager_canary"]["passed"] is False
    assert result["reference_backend"]["passed"] is False


def test_normal_installer_keeps_master_parameters_and_one_step_boundary(monkeypatch):
    import gleipnir.cudnn_fp4_mlp as mlp

    class Qwen3_5MLP(torch.nn.Module):
        pass

    model = torch.nn.Sequential(*[Qwen3_5MLP() for _ in range(32)])
    model.master = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
    original = model.master
    calls = []
    monkeypatch.setattr(
        mlp,
        "install_fp4_mlp",
        lambda model, **kwargs: calls.append(kwargs) or {"modules": list(range(32))},
    )
    marks = []
    monkeypatch.setattr(
        torch.compiler, "cudagraph_mark_step_begin", lambda: marks.append(1)
    )
    installation = native.install_native_fp4_recipe(model)
    assert calls == [{"hardware_packing": True, "fused_descale": True}]
    assert model.master is original and model.master.dtype == torch.float32
    assert installation["cudagraph_step_boundary"] == "physical_model_forward"
    for hook in model._forward_pre_hooks.values():
        hook(model, ())
    assert marks == [1]
    with pytest.raises(ValueError, match="already installed"):
        native.install_graph_step_boundary(model)
    with pytest.raises(ValueError, match="exactly 32"):
        native.install_native_fp4_recipe(torch.nn.Sequential(Qwen3_5MLP()))


def test_environment_retains_existing_caches_and_adds_pinned_overlay(tmp_path):
    original = {"PYTHONPATH": "pinned-fla:fa4", "TRITON_CACHE_DIR": "existing"}
    env = native.native_fp4_environment(original, tmp_path)
    assert original == {"PYTHONPATH": "pinned-fla:fa4", "TRITON_CACHE_DIR": "existing"}
    assert env["PYTHONPATH"].startswith("pinned-fla:fa4:")
    assert env["TRITON_CACHE_DIR"] == "existing"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == str(
        tmp_path / ".cache/training/shared/gpu-0/torchinductor"
    )
    assert env["CUDNN_FRONTEND_COMPILED_CACHE"] == str(
        tmp_path / ".cache/training/shared/cudnn_frontend"
    )
