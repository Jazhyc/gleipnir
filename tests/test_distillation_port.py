import pytest
import torch

from experiments.deception_distillation.build_soft_teacher_cache import (
    build_binary_soft_targets,
)
from experiments.deception_distillation.train_student_sft import (
    CompletionOnlyCollator,
    pairwise_logistic_loss,
    parameter_counts,
    soft_binary_distillation_loss,
)


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
