"""Reject cached comparisons with different membership, labels or inputs."""

import copy

import pytest

from experiments.teacher_disagreement_ood.prepare import bind_cached
from experiments.teacher_disagreement_ood.run import OODContext
from gleipnir.data.monitoring import digest, file_hash, write_json


def example():
    row = {
        "id": "a",
        "prompt": "frozen trajectory",
        "metadata": {"ground_truth": 1, "source_dataset": "source"},
    }
    value = {
        "id": "a",
        "label": 1,
        "source": "source",
        "score": 0.8,
        "source_prompt_sha256": digest(row["prompt"]),
    }
    return row, value


def test_cached_reference_keeps_scores_and_binds_canonical_metadata():
    row, value = example()
    bound = bind_cached([value], [row])
    assert bound[0]["score"] == 0.8
    assert bound[0]["ground_truth"] == 1
    assert bound[0]["source_dataset"] == "source"
    assert "ground_truth" not in value


@pytest.mark.parametrize(
    "key,value",
    [
        ("id", "b"),
        ("label", 0),
        ("source", "other"),
        ("source_prompt_sha256", "different"),
        ("score", float("nan")),
    ],
)
def test_cached_reference_rejects_changed_identity_or_invalid_score(key, value):
    row, original = example()
    changed = copy.deepcopy(original)
    changed[key] = value
    with pytest.raises(ValueError):
        bind_cached([changed], [row])


def test_cached_reference_rejects_duplicates_and_missing_rows():
    row, value = example()
    with pytest.raises(ValueError):
        bind_cached([], [row])
    with pytest.raises(ValueError):
        bind_cached([value, value], [row, row])


def test_ood_context_binds_inputs_without_touching_original_adapter(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}")
    data = tmp_path / "data/followup"
    write_json(
        data / "manifest.json",
        {"files_sha256": {"source.json": file_hash(source)}, "source_sha256": {}},
    )
    ctx = OODContext(
        tmp_path,
        data / "config.yaml",
        {"campaign_id": "followup", "parent_campaign": "results/original"},
    )
    assert ctx.splits == ("ood",)
    assert ctx.adapter == tmp_path / "results/original/4b/monitor"
    ctx.check()
    source.write_text("changed")
    with pytest.raises(ValueError, match="drift"):
        ctx.check()
