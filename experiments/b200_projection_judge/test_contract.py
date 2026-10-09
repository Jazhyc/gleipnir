"""Preference identities and the B-on-tie convention must survive scoring."""

import hashlib

import pytest

from experiments.b200_projection_judge.run import bind, probability


def test_preference_binding_rejects_reordered_labels():
    workload = [
        {
            "id": "x",
            "prompt": "abc",
            "prompt_sha256": hashlib.sha256(b"abc").hexdigest(),
        }
    ]
    with pytest.raises(ValueError, match="identity"):
        bind(workload, [{"id": "y", "label": 0}])


def test_probability_uses_ab_margin_and_preserves_ties():
    assert probability([24.0, 24.0]) == 0.5
    assert probability([0.0, 1000.0]) == 1
    assert probability([1000.0, 0.0]) == 0
    with pytest.raises(ValueError, match="invalid"):
        probability([0.0, float("nan")])
