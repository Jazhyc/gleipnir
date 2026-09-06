import hashlib

import pytest

from gleipnir.prefix_sampling import attach_sampled_prefix, sample_parent_prefixes


def test_sampling_is_order_independent_and_retains_parent_scope():
    candidates = {"b": ["b2", "b1"], "a": ["a1"], "empty": []}
    selected = sample_parent_prefixes(candidates, seed=0, epoch=0)
    reordered = {"empty": [], "a": ["a1"], "b": ["b1", "b2"]}
    assert selected == sample_parent_prefixes(reordered, seed=0, epoch=0)
    assert selected["a"] == "a1"
    assert selected["b"] in candidates["b"]
    assert "empty" not in selected


def test_sampling_does_not_depend_on_other_parents():
    candidates = {"a": ["a1", "a2", "a3"]}
    expected = sample_parent_prefixes(candidates, seed=7, epoch=1)
    candidates["b"] = ["b1"]
    assert sample_parent_prefixes(candidates, seed=7, epoch=1)["a"] == expected["a"]
    draws = {
        sample_parent_prefixes(candidates, seed=7, epoch=epoch)["a"]
        for epoch in range(100)
    }
    assert draws == {"a1", "a2", "a3"}


@pytest.mark.parametrize("candidates", [{"a": ["x", "x"]}, {"a": ["x"], "b": ["x"]}])
def test_sampling_rejects_ambiguous_ids(candidates):
    with pytest.raises(ValueError, match="unique"):
        sample_parent_prefixes(candidates, seed=0, epoch=0)


def test_sampling_rejects_negative_epoch():
    with pytest.raises(ValueError, match="epoch"):
        sample_parent_prefixes({}, seed=0, epoch=-1)


def test_attaching_prefix_preserves_full_supervision_and_excludes_future():
    parent = {
        "prompt_id": "p",
        "student_prompt": (
            "rubric\n<agent_trajectory>\naction\nfuture\n</agent_trajectory>\n"
        ),
        "student_target": "Kimi target",
        "_soft_target": 0.9,
    }
    cached = {
        "id": "p:7",
        "parent_prompt_id": "p",
        "end_character": 7,
        "prefix_sha256": hashlib.sha256(b"action\n").hexdigest(),
        "logprob_0": -1.0,
        "logprob_1": -1.0,
        "score": 0.5,
        "contract_sha256": "contract",
    }
    result = attach_sampled_prefix(parent, cached)
    assert all(result[k] == value for k, value in parent.items())
    assert "future" not in result["prefix_student_prompt"]
    assert result["prefix_soft_target"] == 0.5
    assert "prefix_soft_target" not in parent
    assert attach_sampled_prefix(parent, None) == parent
    for field, value in [
        ("parent_prompt_id", "wrong"),
        ("end_character", 999),
        ("score", 0.8),
    ]:
        with pytest.raises(ValueError):
            attach_sampled_prefix(parent, {**cached, field: value})
