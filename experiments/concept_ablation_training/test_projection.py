"""The intervention projects activations and backward gradients, then disappears."""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from gleipnir.data.monitoring import file_hash
from gleipnir.training.concept_ablation import install, project_residual


def test_projection_autograd_matches_orthogonal_jacobian() -> None:
    unit = torch.tensor([0.6, 0.8])
    hidden = torch.tensor([[2.0, 3.0]], requires_grad=True)
    upstream = torch.tensor([[5.0, -1.0]])
    value = project_residual(hidden, unit)
    (value * upstream).sum().backward()
    expected = upstream - (upstream * unit).sum(-1, keepdim=True) * unit
    torch.testing.assert_close(hidden.grad, expected)
    torch.testing.assert_close(
        (value * unit).sum(), torch.tensor(0.0), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        (hidden.grad * unit).sum(), torch.tensor(0.0), atol=1e-6, rtol=0
    )


def test_training_only_hooks_preserve_export_and_cleanup(tmp_path) -> None:
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(32)])
    model.model.layers[0].forward = torch.compile(
        model.model.layers[0].forward, backend="eager"
    )
    model.config = SimpleNamespace(hidden_size=2)
    path = tmp_path / "direction.npz"
    np.savez(path, unit=np.array([1.0, 0.0], np.float32))
    before = tuple(model.state_dict())
    metadata, handles = install(model, path, file_hash(path))
    x = torch.tensor([[1.0, 2.0]])
    y = model.model.layers[0](x)
    assert y[0, 0] == 0 and metadata["calls"]["0"] == 1
    y.sum().backward()
    assert torch.count_nonzero(model.model.layers[0].weight.grad[0]) == 0
    model.eval()
    torch.testing.assert_close(
        model.model.layers[0](x), model.model.layers[0].forward(x)
    )
    for handle in handles:
        handle.remove()
    model.train()
    torch.testing.assert_close(
        model.model.layers[0](x), model.model.layers[0].forward(x)
    )
    assert tuple(model.state_dict()) == before


def test_campaign_routes_frozen_direction_to_training_command() -> None:
    from pathlib import Path

    from gleipnir.campaigns.monitoring.contract import Campaign
    from gleipnir.campaigns.training_command import training_command

    root = Path(__file__).resolve().parents[2]
    ctx = Campaign.load(root, Path(__file__).with_name("config.yaml"))
    job = ctx.job()
    assert job["concept_ablation_path"] == str(ctx.input("concept_direction"))
    command = training_command(job)
    assert (
        f"++student.training.concept_ablation_path={job['concept_ablation_path']}"
        in command
    )
    assert (
        f"++student.training.concept_ablation_sha256={job['concept_ablation_sha256']}"
        in command
    )
