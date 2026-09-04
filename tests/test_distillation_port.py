import pytest
import torch

from experiments.deception_distillation.build_soft_teacher_cache import (
    build_binary_soft_targets,
)
from experiments.deception_distillation.train_student_sft import (
    CompletionOnlyCollator,
    GroupDROLoss,
    causal_conv1d_kernel_modules,
    collate_eva_features,
    dataset_sampling_weights,
    forward_final_token_hidden,
    forward_final_token_logits,
    forward_final_token_logits_and_head_inputs,
    gated_delta_kernel_modules,
    pairwise_logistic_loss,
    parameter_counts,
    soft_binary_distillation_loss,
    soft_binary_distillation_losses,
    straight_through_decision_logits,
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


class HiddenStateModel(torch.nn.Module):
    def __init__(self, hidden_states: torch.Tensor) -> None:
        super().__init__()
        self.model = self
        self.hidden_states = hidden_states

    def get_base_model(self):
        return self

    def forward(self, input_ids, attention_mask, use_cache):
        del input_ids, attention_mask, use_cache
        return type(
            "Output", (), {"last_hidden_state": self.hidden_states}
        )()


class HiddenProjectingModel(torch.nn.Module):
    def __init__(self, hidden_states: torch.Tensor) -> None:
        super().__init__()
        self.hidden_states = hidden_states
        self.lm_head = torch.nn.Linear(hidden_states.shape[-1], 5, bias=False)

    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep=None):
        del input_ids, attention_mask, use_cache
        selected = self.hidden_states[:, logits_to_keep, :]
        logits = self.lm_head(selected)
        return type("Output", (), {"logits": logits})()


class DictLikeTokenizer:
    def pad(self, features, *, padding, return_tensors):
        assert features == [{"input_ids": [1]}]
        assert padding is True
        assert return_tensors == "pt"

        class BatchEncoding(dict):
            pass

        return BatchEncoding(input_ids=torch.tensor([[1]]))


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
    assert collator.padding_statistics() == {
        "batches": 1,
        "examples": 2,
        "input_tokens": 3,
        "input_padded_tokens": 4,
        "input_padding_fraction": 0.25,
        "direct_tokens": 0,
        "direct_padded_tokens": 0,
        "direct_padding_fraction": None,
    }


def test_eva_collator_returns_a_concrete_dictionary() -> None:
    batch = collate_eva_features(DictLikeTokenizer(), [{"input_ids": [1]}])

    assert type(batch) is dict
    assert batch["input_ids"].tolist() == [[1]]


def test_decision_head_straight_through_uses_head_forward_and_token_gradient() -> None:
    head_logits = torch.randn(4, 2, requires_grad=True)
    token_logits = torch.randn(4, 2, requires_grad=True)

    combined = straight_through_decision_logits(head_logits, token_logits)
    combined.sum().backward()

    assert torch.equal(combined, head_logits)
    assert torch.equal(head_logits.grad, torch.ones_like(head_logits))
    assert torch.equal(token_logits.grad, torch.ones_like(token_logits))


def test_soft_and_pairwise_losses_reward_teacher_ordering() -> None:
    logits = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    targets = torch.tensor([0.1, 0.9])
    good = soft_binary_distillation_loss(logits, targets, loss_type="bce")
    bad = soft_binary_distillation_loss(-logits, targets, loss_type="bce")
    assert good < bad
    row_losses = soft_binary_distillation_losses(logits, targets, loss_type="bce")
    assert row_losses.shape == (2,)
    assert row_losses.mean() == pytest.approx(good)

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
        "auxiliary_trainable_parameters": 0,
    }

    full_model = torch.nn.Linear(3, 2)
    counts = parameter_counts(full_model, "full")
    assert counts["total_parameters"] == 8
    assert counts["trainable_parameters"] == 8
    assert counts["lora_trainable_parameters"] == 0
    assert counts["auxiliary_trainable_parameters"] == 0


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


def test_final_hidden_states_use_the_same_unique_position_gather() -> None:
    hidden_states = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    input_ids = torch.ones((2, 4), dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    selected, _ = forward_final_token_hidden(
        HiddenStateModel(hidden_states), input_ids, attention_mask
    )

    expected = torch.stack((hidden_states[0, 2], hidden_states[1, 1]))
    assert torch.equal(selected, expected)


def test_selected_logits_capture_the_exact_lm_head_inputs() -> None:
    hidden_states = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    model = HiddenProjectingModel(hidden_states)
    input_ids = torch.ones((2, 4), dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    _, selected, _ = forward_final_token_logits_and_head_inputs(
        model, input_ids, attention_mask, "selected_positions"
    )

    expected = torch.stack((hidden_states[0, 2], hidden_states[1, 1]))
    assert torch.equal(selected, expected)


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


def test_causal_conv1d_kernel_modules_report_fast_and_fallback_paths() -> None:
    model = torch.nn.Module()
    fast_child = torch.nn.Module()
    fallback_child = torch.nn.Module()

    def fake_kernel():
        return None

    fake_kernel.__module__ = "causal_conv1d.causal_conv1d_interface"
    fast_child.causal_conv1d_fn = fake_kernel
    fallback_child.causal_conv1d_fn = None
    model.add_module("fast", fast_child)
    model.add_module("fallback", fallback_child)

    assert causal_conv1d_kernel_modules(model) == [
        "causal_conv1d.causal_conv1d_interface",
        "torch_fallback",
    ]


def test_dataset_sampling_weights_match_declared_dataset_mass() -> None:
    dataset_ids = [0, 0, 0, 0, 1]

    proportional, proportional_mass = dataset_sampling_weights(
        dataset_ids, "proportional"
    )
    square_root, square_root_mass = dataset_sampling_weights(
        dataset_ids, "sqrt_balanced"
    )
    uniform, uniform_mass = dataset_sampling_weights(
        dataset_ids, "uniform_dataset"
    )

    assert proportional.tolist() == [1.0] * 5
    assert proportional_mass == pytest.approx({0: 0.8, 1: 0.2})
    assert square_root_mass == pytest.approx({0: 2 / 3, 1: 1 / 3})
    assert uniform_mass == pytest.approx({0: 0.5, 1: 0.5})
    assert uniform.tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25, 1.0])
    assert square_root[0] < proportional[0]

    with pytest.raises(ValueError, match="unknown dataset sampling"):
        dataset_sampling_weights(dataset_ids, "invalid")


def test_group_dro_upweights_the_higher_loss_group() -> None:
    group_dro = GroupDROLoss(2, eta=2.0, ema=0.0)
    losses = torch.tensor([0.1, 0.2, 1.0, 1.2])
    dataset_ids = torch.tensor([0, 0, 1, 1])

    weighted = group_dro(losses, dataset_ids)
    snapshot = group_dro.snapshot()

    assert weighted > losses.mean()
    assert snapshot is not None
    assert snapshot["weights"][1] > snapshot["weights"][0]
