import pytest
import torch
from transformers import Qwen3_5TextConfig

from gleipnir.branch_training import BranchCache, plan_branches


def test_plan_preserves_requests_and_avoids_single_token_segments():
    requests = [[1, 2, 3, 9, 9], [1, 2, 3, 4, 9, 9], [1, 2, 3, 4, 5, 6, 9, 9]]
    plan = plan_branches(requests)
    previous = 0
    for split in sorted(set(plan.split_positions)):
        assert split == previous or split - previous >= 2
        previous = split
    for row, split in zip(plan.requests, plan.split_positions, strict=True):
        assert plan.requests[-1][:split] + row[split:] == row
        assert len(row) - split >= 2
    assert plan.processed_tokens < sum(map(len, requests))
    with pytest.raises(ValueError):
        plan_branches([[1]])


def test_tiny_qwen_scores_and_gradients():
    from experiments.monitoring_branching.run import canary

    assert canary()["passed"]


def test_fork_preserves_shared_gradients_without_overwriting_parent():
    config = Qwen3_5TextConfig(
        num_hidden_layers=2, layer_types=["linear_attention", "full_attention"]
    )
    parent = BranchCache(config=config)
    conv = torch.randn(1, 2, 4, requires_grad=True)
    recurrent = torch.randn(1, 2, 4, 4, requires_grad=True)
    keys = torch.randn(1, 2, 3, 4, requires_grad=True)
    values = torch.randn(1, 2, 3, 4, requires_grad=True)
    parent.update_conv_state(conv, 0)
    parent.update_recurrent_state(recurrent, 0)
    parent.update(keys, values, 1)
    branch = parent.fork()
    branch.update_conv_state(conv * 2, 0)
    branch.update_recurrent_state(recurrent * 3, 0)
    branch.update(keys * 4, values * 5, 1)
    assert parent.layers[0].conv_states[0] is conv
    assert parent.layers[0].recurrent_states[0] is recurrent
    assert parent.get_seq_length() == 3
    assert branch.get_seq_length() == 6
    (
        branch.layers[0].conv_states[0].sum()
        + branch.layers[0].recurrent_states[0].sum()
        + branch.layers[1].keys.sum()
        + branch.layers[1].values.sum()
    ).backward()
    torch.testing.assert_close(conv.grad, torch.full_like(conv, 2))
    torch.testing.assert_close(recurrent.grad, torch.full_like(recurrent, 3))
    torch.testing.assert_close(keys.grad, torch.full_like(keys, 5))
    torch.testing.assert_close(values.grad, torch.full_like(values, 6))
