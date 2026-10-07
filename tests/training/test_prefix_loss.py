import pytest
import torch

from gleipnir.prefix_loss import trajectory_prefix_loss


def test_parent_mass_and_gradients():
    full = torch.tensor([2.0, 4.0, 6.0], requires_grad=True)
    prefixes = torch.tensor([1.0, 3.0, 5.0], requires_grad=True)
    parents = torch.tensor([0, 0, 1])
    loss = trajectory_prefix_loss(full, prefixes, parents, 0.5)
    assert loss.item() == pytest.approx((2 + 6.5 / 1.5 + 6) / 3)
    loss.backward()
    assert full.grad.tolist() == pytest.approx([1 / 4.5, 1 / 4.5, 1 / 3])
    assert prefixes.grad.tolist() == pytest.approx([1 / 18, 1 / 18, 1 / 9])


def test_duplicating_prefixes_does_not_change_parent_weight():
    full = torch.tensor([1.0, 2.0])
    a = trajectory_prefix_loss(
        full, torch.tensor([3.0, 4.0]), torch.tensor([0, 1]), 0.25
    )
    b = trajectory_prefix_loss(
        full, torch.tensor([3.0, 3.0, 4.0]), torch.tensor([0, 0, 1]), 0.25
    )
    assert a.item() == b.item()


def test_no_prefixes_and_zero_weight():
    full = torch.tensor([1.0, 3.0])
    assert (
        trajectory_prefix_loss(
            full, torch.tensor([]), torch.tensor([], dtype=torch.long), 0.5
        )
        == 2
    )
    assert (
        trajectory_prefix_loss(full, torch.tensor([100.0]), torch.tensor([0]), 0) == 2
    )


def test_invalid_parent_rejected():
    with pytest.raises(ValueError, match="out of range"):
        trajectory_prefix_loss(
            torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([1]), 0.5
        )
