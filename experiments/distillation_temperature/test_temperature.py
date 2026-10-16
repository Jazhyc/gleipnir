from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.deception_distillation.train_student_sft import (
    soft_binary_distillation_losses,
)
from experiments.distillation_temperature.audit import kd_weights
from gleipnir.campaigns.training_command import training_command
from gleipnir.training.validation import validate_training_metadata

ROOT = Path(__file__).resolve().parents[2]


def test_unit_temperature_is_bitwise_the_original_bce() -> None:
    logits = torch.tensor([[0.5, 2.0], [1.0, -3.0], [0.0, 0.25]])
    targets = torch.tensor([0.993, 0.004, 0.6])
    original = F.binary_cross_entropy_with_logits(
        logits[:, 1] - logits[:, 0], targets, reduction="none"
    )
    assert torch.equal(soft_binary_distillation_losses(logits, targets), original)
    assert torch.equal(
        soft_binary_distillation_losses(logits, targets, kd_temperature=1.0), original
    )


def test_kd_temperature_scales_both_sides_and_keeps_the_optimum() -> None:
    targets = torch.tensor([0.993, 0.004, 0.6])
    teacher = torch.logit(targets)
    logits = torch.stack([torch.zeros(3), teacher + 0.3], dim=1).requires_grad_()
    losses = soft_binary_distillation_losses(logits, targets, kd_temperature=4.0)
    expected = 16 * F.binary_cross_entropy_with_logits(
        (teacher + 0.3) / 4, torch.sigmoid(teacher / 4), reduction="none"
    )
    assert torch.allclose(losses, expected)
    losses.sum().backward()
    grad = logits.grad[:, 1]
    # Small-error gradient matches sigmoid'(z_t / T): confident rows gain weight.
    weights = torch.from_numpy(kd_weights(teacher.double().numpy(), 4.0)).float()
    assert torch.allclose(grad, 0.3 * weights, rtol=0.1)
    at_teacher = torch.stack([torch.zeros(3), teacher], dim=1).requires_grad_()
    soft_binary_distillation_losses(
        at_teacher, targets, kd_temperature=4.0
    ).sum().backward()
    assert torch.allclose(at_teacher.grad, torch.zeros(3, 2), atol=1e-6)


def test_teacher_only_scale_sharpens_targets() -> None:
    targets = torch.tensor([0.9])
    logits = torch.zeros(1, 2)
    sharpened = soft_binary_distillation_losses(logits, targets, target_logit_scale=0.5)
    expected = F.binary_cross_entropy_with_logits(
        torch.zeros(1), torch.sigmoid(torch.logit(targets) / 0.5), reduction="none"
    )
    assert torch.allclose(sharpened, expected)


def test_kd_temperature_rejects_huber_and_nonpositive_values() -> None:
    logits, targets = torch.zeros(1, 2), torch.tensor([0.5])
    with pytest.raises(ValueError, match="requires the BCE"):
        soft_binary_distillation_losses(
            logits, targets, loss_type="huber", kd_temperature=2.0
        )
    with pytest.raises(ValueError, match="finite and positive"):
        soft_binary_distillation_losses(logits, targets, kd_temperature=0.0)


def test_campaign_command_renders_and_validation_checks_temperatures() -> None:
    with initialize_config_dir(
        config_dir=str(ROOT / "src/gleipnir/configs/systems_screen"),
        version_base=None,
    ):
        recipe = OmegaConf.to_container(
            compose(config_name="qwen35_4b_b200_packed_sdpa"), resolve=True
        )["recipe"]
    job = {
        **recipe,
        "job_name": "kd-test",
        "output_dir": "results/kd-test",
        "student_rows": "data/test.jsonl",
        "soft_targets": "data/soft.jsonl",
        "causal_adapter_dir": "results/kd-test/causal_adapter",
        "max_steps": -1,
        "soft_kd_temperature": 2.0,
        "soft_target_logit_scale": 0.5,
    }
    command = training_command(job)
    assert "++student.training.soft_kd_temperature=2.0" in command
    assert "++student.training.soft_target_logit_scale=0.5" in command
    with initialize_config_dir(
        config_dir=str(ROOT / "experiments/tool_trajectory_monitoring"),
        version_base=None,
    ):
        cfg = compose(config_name="distillation_config", overrides=command[6:])
    assert cfg.student.training.soft_kd_temperature == 2.0
    assert cfg.student.training.soft_target_logit_scale == 0.5
    baseline = {
        k: v
        for k, v in job.items()
        if k not in {"soft_kd_temperature", "soft_target_logit_scale"}
    }
    assert not any("soft_kd" in arg for arg in training_command(baseline))


def test_validation_rejects_a_temperature_mismatch() -> None:
    metadata = {
        "training_state": {"global_step": 1},
        "sequence_packing": {
            "initial_master_sha256": "a",
            "final_master_sha256": "b",
            **{
                k: {"passed": True}
                for k in ("eager_canary", "compiled_canary", "preflight")
            },
        },
        "losses": {
            "soft_weight": 1.0,
            "direct_weight": 0.0,
            "completion_weight": 0,
            "accumulation_policy": "sum_per_example_over_logical_batch_v1",
            "soft_kd_temperature": 1.0,
        },
    }
    job = {
        "soft_loss_weight": 1.0,
        "direct_loss_weight": 0.0,
        "soft_kd_temperature": 2.0,
    }
    profile = {"metadata_expectations": {}}
    with pytest.raises(ValueError, match="objective"):
        validate_training_metadata(
            metadata, job, "a", expected_steps=1, profile=profile
        )


def test_arms_change_only_identity_contract_text_and_one_knob() -> None:
    import yaml

    from experiments.distillation_temperature.prepare import ARMS, CONTROL, arm_config

    control = yaml.safe_load((ROOT / "data" / CONTROL / "config.yaml").read_text())
    allowed = {"campaign_id", "extra_sources", "gpu_uuid", "merged_model", "inputs"}
    allowed |= {"hypothesis", "intervention", "selection"}
    for name, knob in ARMS.items():
        arm = arm_config(control, name, knob, "GPU-test", "selection-sha")
        changed = {k for k in set(arm) | set(control) if arm.get(k) != control.get(k)}
        assert changed == allowed | set(knob)
        assert {
            k for k in arm["inputs"] if arm["inputs"][k] != control["inputs"][k]
        } == {"serving_selection"}
