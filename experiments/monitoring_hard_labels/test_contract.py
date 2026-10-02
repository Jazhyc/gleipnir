"""Guard hard-label provenance, objective scale, packed launch and ID selection."""

import copy

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.monitoring_hard_labels.prepare import (
    FRACTIONS,
    ROOT,
    configuration,
    validate_holdout,
)
from experiments.monitoring_hard_labels.summarize import select_winner
from experiments.monitoring_hard_labels.train import make_job
from gleipnir.monitoring_training_command import training_command
from gleipnir.packed_training import validate_packed_training_config


def test_four_cells_use_matched_packed_recipe_and_normalized_source_labels():
    config = configuration()
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/gleipnir/configs/systems_screen")
    ):
        recipe = OmegaConf.to_container(
            compose(config_name=config["profile"]), resolve=True
        )["recipe"]
    common = None
    for variant, fraction in FRACTIONS.items():
        job = make_job(config, recipe, "4b", variant)
        assert job["direct_loss_weight"] == fraction
        assert job["soft_loss_weight"] + job["direct_loss_weight"] == 1
        assert job["learning_rate"] == 2e-5
        assert job["num_train_epochs"] == 1
        assert job["max_steps"] == -1
        command = training_command(job)
        with initialize_config_dir(
            version_base=None,
            config_dir=str(ROOT / "experiments/tool_trajectory_monitoring"),
        ):
            student = OmegaConf.to_container(
                compose(
                    config_name="distillation_config",
                    overrides=command[command.index("--config-name") + 2 :],
                ).student,
                resolve=True,
            )
        assert (
            student["soft_teacher_artifact"] is None
            if fraction == 1
            else student["soft_teacher_artifact"] is not None
        )
        assert validate_packed_training_config(student)
        assert student["quantization"]["enabled"] is False
        assert student["training"]["gradient_checkpointing"] is False
        assert student["training"]["per_device_train_batch_size"] == 32
        assert student["training"]["gradient_accumulation_steps"] == 1
        unchanged = {
            k: v
            for k, v in job.items()
            if k
            not in {
                "job_name",
                "student_rows",
                "output_dir",
                "causal_adapter_dir",
                "model_dir",
                "soft_loss_weight",
                "direct_loss_weight",
            }
        }
        if common is None:
            common = unchanged
        assert unchanged == common


def test_original_lineage_overlap_fails_after_cot_transformation():
    training = [{"trajectory_sha256": "original"}]
    evaluation = [
        {
            "metadata": {
                "trajectory_sha256": "cleaned",
                "original_metadata": {"trajectory_sha256": "original"},
            }
        }
    ]
    with pytest.raises(ValueError, match="lineage overlap"):
        validate_holdout(training, evaluation)
    evaluation[0]["metadata"]["original_metadata"]["trajectory_sha256"] = "other"
    validate_holdout(training, evaluation)


def metric_rows():
    return [
        {
            "variant": v,
            "pauroc_at_20": 0.8,
            "auroc": 0.9,
            "brier": 0.1,
            "sources": {"gloom_exfiltration": 0.7, "test_stride": 0.9},
        }
        for v in FRACTIONS
    ]


def test_selection_retains_control_and_rejects_source_and_calibration_regressions():
    limits = configuration()["selection"]
    rows = metric_rows()
    assert select_winner(copy.deepcopy(rows), limits)["control_retained"]
    rows[1].update(
        pauroc_at_20=0.82, sources={"gloom_exfiltration": 0.74, "test_stride": 0.9}
    )
    assert select_winner(copy.deepcopy(rows), limits)["selected_variant"] == "hard010"
    rows[2].update(
        pauroc_at_20=0.85, sources={"gloom_exfiltration": 0.69, "test_stride": 0.98}
    )
    rows[3].update(pauroc_at_20=0.86, brier=0.11)
    assert select_winner(copy.deepcopy(rows), limits)["selected_variant"] == "hard010"
    with pytest.raises(ValueError, match="four-cell"):
        select_winner(rows[:-1], limits)
    rows[1]["brier"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        select_winner(rows, limits)


def test_source_hard_ce_and_soft_bce_match_convex_target_gradients():
    import torch
    import torch.nn.functional as functional

    logits = torch.tensor([[0.1, -0.3], [1.2, -0.8], [-0.2, 0.6]], requires_grad=True)
    hard = torch.tensor([0, 1, 1])
    soft = torch.tensor([0.35, 0.75, 0.45])
    for fraction in FRACTIONS.values():
        margin = logits[:, 1] - logits[:, 0]
        loss = (1 - fraction) * functional.binary_cross_entropy_with_logits(
            margin, soft
        )
        loss += fraction * functional.cross_entropy(logits, hard)
        mixed = functional.binary_cross_entropy_with_logits(
            margin, (1 - fraction) * soft + fraction * hard.float()
        )
        torch.testing.assert_close(loss, mixed)
        torch.testing.assert_close(
            torch.autograd.grad(loss, logits, retain_graph=True)[0],
            torch.autograd.grad(mixed, logits, retain_graph=True)[0],
        )
