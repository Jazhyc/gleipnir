import torch

from gleipnir.training import MuonAdamW, muon_adamw_param_groups


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
    optimizer = MuonAdamW(groups, lr=1e-3, muon_lr=1e-3)
    loss = model.bias.square().sum()
    loss.backward()
    before_bias = model.bias.detach().clone()
    optimizer.step()
    assert not torch.equal(model.bias, before_bias)
