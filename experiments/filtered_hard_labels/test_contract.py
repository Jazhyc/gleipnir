from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from experiments.filtered_hard_labels.prepare import CONTROL, arm_config

ROOT = Path(__file__).resolve().parents[2]


def test_arm_changes_only_identity_text_and_loss_weights() -> None:
    control = yaml.safe_load((ROOT / "data" / CONTROL / "config.yaml").read_text())
    arm = arm_config(control, "GPU-test", "selection-sha")
    changed = {k for k in set(arm) | set(control) if arm.get(k) != control.get(k)}
    assert changed == {
        "campaign_id",
        "extra_sources",
        "gpu_uuid",
        "merged_model",
        "inputs",
        "hypothesis",
        "intervention",
        "selection",
        "soft_loss_weight",
        "direct_loss_weight",
    }
    assert arm["soft_loss_weight"] + arm["direct_loss_weight"] == 1
    assert arm["learning_rate"] == control["learning_rate"] == 5e-5
    assert {k for k in arm["inputs"] if arm["inputs"][k] != control["inputs"][k]} == {
        "serving_selection"
    }


def test_mixture_equals_bce_against_the_convex_target() -> None:
    logits = torch.tensor([[0.2, 1.1], [0.5, -2.0], [0.0, 0.3]])
    soft = torch.tensor([0.9, 0.02, 0.6])
    hard = torch.tensor([1, 0, 1])
    margin = logits[:, 1] - logits[:, 0]
    mixed = 0.7 * F.binary_cross_entropy_with_logits(margin, soft) + 0.3 * (
        F.cross_entropy(logits, hard)
    )
    convex = F.binary_cross_entropy_with_logits(margin, 0.7 * soft + 0.3 * hard)
    assert torch.allclose(mixed, convex)
