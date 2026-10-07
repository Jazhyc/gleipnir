"""Exercise Transformers' real accumulation path with independent loss gradients."""

import pytest
import torch
from transformers import Trainer, TrainingArguments

from gleipnir.prefix_loss import trajectory_prefix_loss
from gleipnir.training import configure_mean_loss_accumulation


class TwoObjectiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.direct = torch.nn.Parameter(torch.tensor(1.0))
        self.auxiliary = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, **kwargs):
        raise AssertionError("the custom trainer owns both objectives")


class SequentialMeanTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        auxiliary = 0.2 * model.auxiliary
        self.accelerator.backward(auxiliary / self.current_gradient_accumulation_steps)
        return auxiliary.detach() + model.direct


@pytest.mark.parametrize("explicit_mean_loss", [False, True])
@pytest.mark.parametrize("window", [1, 16, 32])
@pytest.mark.parametrize("completion_labels", [False, True])
def test_actual_trainer_scaling_with_completion_labels(
    tmp_path,
    explicit_mean_loss,
    window,
    completion_labels,
):
    model = TwoObjectiveModel()
    trainer = SequentialMeanTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            gradient_accumulation_steps=32,
            report_to="none",
        ),
    )
    assert trainer.model_accepts_loss_kwargs is True
    assert trainer.accelerator.gradient_accumulation_steps == 1
    if explicit_mean_loss:
        configure_mean_loss_accumulation(trainer)
    trainer.current_gradient_accumulation_steps = window
    batches = [
        {"labels": torch.tensor([[-100, 1]])}
        if completion_labels
        else {"direct_input_ids": torch.tensor([[1]])}
        for _ in range(window)
    ]
    items = trainer._get_num_items_in_batch(batches, torch.device("cpu"))
    for batch in batches:
        trainer.training_step(model, batch, num_items_in_batch=items)
    assert model.auxiliary.grad.item() == pytest.approx(0.2)
    expected = 1.0 if explicit_mean_loss or not completion_labels else float(window)
    assert model.direct.grad.item() == pytest.approx(expected)


@pytest.mark.parametrize("weight", [0.25, 0.5])
@pytest.mark.parametrize("window", [1, 3, 16, 32])
def test_sequential_prefix_gradients_match_parent_objective(tmp_path, weight, window):
    class PrefixTrainer(Trainer):
        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            # Three parents, only parent 0 and 2 have intermediate targets.
            prefix_losses = torch.stack([model.auxiliary**2, 2 * model.auxiliary**2])
            term = weight / (1 + weight) * prefix_losses.sum() / 3
            self.accelerator.backward(term / self.current_gradient_accumulation_steps)
            full_losses = torch.stack(
                [model.direct**2, 2 * model.direct**2, 3 * model.direct**2]
            )
            scales = torch.tensor([1 / (1 + weight), 1, 1 / (1 + weight)])
            return term.detach() + (scales * full_losses).mean()

    model = TwoObjectiveModel()
    reference = TwoObjectiveModel()
    expected_loss = trajectory_prefix_loss(
        torch.stack(
            [reference.direct**2, 2 * reference.direct**2, 3 * reference.direct**2]
        ),
        torch.stack([reference.auxiliary**2, 2 * reference.auxiliary**2]),
        torch.tensor([0, 2]),
        weight,
    )
    expected_loss.backward()
    trainer = PrefixTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            gradient_accumulation_steps=32,
            report_to="none",
        ),
    )
    configure_mean_loss_accumulation(trainer)
    trainer.current_gradient_accumulation_steps = window
    for _ in range(window):
        trainer.training_step(model, {"direct_input_ids": torch.tensor([[1]])})
    assert model.direct.grad.item() == pytest.approx(reference.direct.grad.item())
    assert model.auxiliary.grad.item() == pytest.approx(reference.auxiliary.grad.item())
