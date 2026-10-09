"""Auxiliary judging must select A/B and retain full-vocabulary normalization."""

import pytest
import torch

from gleipnir.serving.lens_readout import judge_readout


def test_ab_readout_is_distinct_from_monitor_and_full_vocabulary_mass():
    embedding = torch.zeros(40, 2)
    embedding[32], embedding[33] = torch.tensor([2.0, 0.0]), torch.tensor([0.0, 3.0])
    embedding[15] = torch.tensor([8.0, 8.0])
    hidden = torch.tensor([[1.0, 2.0]])
    logits = torch.nn.functional.linear(hidden, embedding)[0]
    value = judge_readout(logits, hidden, embedding)
    assert value["token_ids"] == [32, 33]
    assert value["logits"] == value["fp32_head_logits"] == [2.0, 6.0]
    expected = torch.softmax(logits, dim=0)[[32, 33]].sum().item()
    assert value["probability_mass"] == pytest.approx(expected, rel=1e-5)
    assert value["probability_mass"] < 0.001


def test_ab_nonfinite_head_is_rejected():
    embedding = torch.zeros(40, 2)
    embedding[32, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        judge_readout(torch.zeros(40), torch.ones(1, 2), embedding)
