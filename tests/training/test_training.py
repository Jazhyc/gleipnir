import pytest
import torch

from gleipnir.training import (
    MuonAdamW,
    muon_adamw_param_groups,
    muon_update_scale,
)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = torch.nn.Parameter(torch.ones(2, 2))
        self.bias = torch.nn.Parameter(torch.ones(2))


def test_parameter_groups_route_matrices_to_muon() -> None:
    groups = muon_adamw_param_groups(TinyModel(), weight_decay=0.1)
    assert [group["algorithm"] for group in groups] == ["muon", "adamw"]
    assert groups[1]["weight_decay"] == 0.0


def test_muon_adamw_updates_trainable_parameters() -> None:
    model = TinyModel()
    groups = [
        {
            "params": [model.bias],
            "weight_decay": 0.0,
            "algorithm": "adamw",
        }
    ]
    optimizer = MuonAdamW(groups, lr=1e-3)
    loss = model.bias.square().sum()
    loss.backward()
    before_bias = model.bias.detach().clone()
    optimizer.step()
    assert not torch.equal(model.bias, before_bias)


def test_muon_matches_adamw_update_rms_per_matrix_shape() -> None:
    wide = torch.empty(8, 128)
    tall = torch.empty(128, 8)

    assert muon_update_scale(wide) == pytest.approx(0.2 * 128**0.5)
    assert muon_update_scale(tall) == pytest.approx(0.2 * 128**0.5)
    assert muon_update_scale(wide, "original") == pytest.approx(1.0)
    assert muon_update_scale(tall, "original") == pytest.approx(4.0)


def test_muon_reads_scheduler_managed_learning_rate() -> None:
    first = torch.nn.Parameter(torch.eye(2))
    second = torch.nn.Parameter(torch.eye(2))
    first.grad = torch.ones_like(first)
    second.grad = torch.ones_like(second)
    full = MuonAdamW([{"params": [first], "algorithm": "muon"}], lr=1e-3)
    half = MuonAdamW([{"params": [second], "algorithm": "muon"}], lr=1e-3)
    half.param_groups[0]["lr"] = 5e-4

    full.step()
    half.step()

    full_delta = float((torch.eye(2) - first).norm().detach())
    half_delta = float((torch.eye(2) - second).norm().detach())
    assert half_delta == pytest.approx(full_delta / 2, rel=1e-3)
