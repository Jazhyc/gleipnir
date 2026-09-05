"""Exercise Transformers' real accumulation path with independent loss gradients."""

import pytest
import torch
from transformers import Trainer, TrainingArguments

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
