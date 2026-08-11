import pytest
import torch

from experiments.deception_distillation.build_soft_teacher_cache import (
    build_binary_soft_targets,
)
from experiments.deception_distillation.train_student_sft import (
    CompletionOnlyCollator,
    forward_final_token_logits,
    gated_delta_kernel_modules,
    pairwise_logistic_loss,
    parameter_counts,
    soft_binary_distillation_loss,
)


class PositionSelectingModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.requested_positions = None

    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep=None):
        del input_ids, attention_mask, use_cache
        self.requested_positions = logits_to_keep
        selected = self.logits
        if logits_to_keep is not None:
            selected = selected[:, logits_to_keep, :]
        return type("Output", (), {"logits": selected})()


def test_binary_teacher_probability_is_preserved() -> None:
    rows = build_binary_soft_targets(
        [
            {
                "dataset": "source-a",
                "index": "1",
                "label": 0,
                "score": 0.73,
                "target_probs": {"negative": 0.27, "positive": 0.73},
                "missing_rating_token_ids": [],
                "parse_error": False,
            }
        ]
    )
    assert rows[0]["soft_target"] == pytest.approx(0.73)


def test_completion_collator_masks_prompt_padding() -> None:
    collator = CompletionOnlyCollator(pad_token_id=9)
    batch = collator(
        [
            {"input_ids": [1, 2], "labels": [-100, 2]},
            {"input_ids": [3], "labels": [3]},
        ]
    )
    assert batch["input_ids"].tolist() == [[1, 2], [3, 9]]
    assert batch["labels"].tolist() == [[-100, 2], [3, -100]]


def test_soft_and_pairwise_losses_reward_teacher_ordering() -> None:
    logits = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    targets = torch.tensor([0.1, 0.9])
    good = soft_binary_distillation_loss(logits, targets, loss_type="bce")
    bad = soft_binary_distillation_loss(-logits, targets, loss_type="bce")
    assert good < bad

    margins = logits[:, 1] - logits[:, 0]
    labels = torch.tensor([0, 1])
    dataset_ids = torch.tensor([0, 0])
    assert pairwise_logistic_loss(margins, labels, dataset_ids, 1.0) < (
        pairwise_logistic_loss(-margins, labels, dataset_ids, 1.0)
    )


def test_parameter_counts_validate_lora_and_full_layouts() -> None:
    lora_model = torch.nn.Module()
    lora_model.register_parameter(
        "frozen", torch.nn.Parameter(torch.ones(2), requires_grad=False)
    )
    lora_model.register_parameter("lora_A", torch.nn.Parameter(torch.ones(3)))
    assert parameter_counts(lora_model, "lora") == {
        "total_parameters": 5,
        "trainable_parameters": 3,
        "lora_trainable_parameters": 3,
    }

    full_model = torch.nn.Linear(3, 2)
    counts = parameter_counts(full_model, "full")
    assert counts["total_parameters"] == 8
    assert counts["trainable_parameters"] == 8
    assert counts["lora_trainable_parameters"] == 0


def test_selected_position_logits_match_full_logits_with_right_padding() -> None:
    logits = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    input_ids = torch.ones((2, 4), dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    full_model = PositionSelectingModel(logits)
    full, _ = forward_final_token_logits(
        full_model, input_ids, attention_mask, "full"
    )
    selected_model = PositionSelectingModel(logits)
    selected, _ = forward_final_token_logits(
        selected_model, input_ids, attention_mask, "selected_positions"
    )

    assert torch.equal(selected, full)
    assert torch.equal(selected_model.requested_positions, torch.tensor([1, 2]))


def test_gated_delta_kernel_modules_report_bound_implementation() -> None:
    model = torch.nn.Module()
    child = torch.nn.Module()

    def fake_kernel():
        return None

    fake_kernel.__module__ = "fla.ops.gated_delta_rule.chunk"
    child.chunk_gated_delta_rule = fake_kernel
    model.add_module("gated_delta", child)

    assert gated_delta_kernel_modules(model) == [
        "fla.ops.gated_delta_rule.chunk"
    ]
