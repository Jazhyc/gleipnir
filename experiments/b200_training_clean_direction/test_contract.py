"""Original-data joins must reject synthetic views and stale source lineage."""

import pytest

from experiments.b200_training_clean_direction.prepare import join_originals


def records():
    clean = {
        "dataset": "x",
        "index": "a",
        "trajectory_sha256": "trajectory",
        "label": 0,
        "teacher_rendered_prompt_sha256": "teacher",
    }
    teacher = {
        "dataset": "x",
        "index": "a",
        "rendered_prompt_sha256": "teacher",
        "soft_target": 0.2,
    }
    census = {"id": "a", "trajectory_sha256": "trajectory", "label": 0, "score": 0.3}
    return clean, teacher, census


def test_exact_original_join_keeps_teacher_and_census_separate():
    a, b, c = records()
    result = join_originals([a], [b], [c])
    assert result[0] == a | {"soft_target": 0.2, "census_score": 0.3}
    assert "soft_target" not in a


@pytest.mark.parametrize(
    "change,match",
    [
        ({"augmentation": {}}, "synthetic"),
        ({"trajectory_sha256": "changed"}, "census"),
        ({"teacher_rendered_prompt_sha256": "changed"}, "teacher"),
    ],
)
def test_invalid_original_or_lineage_fails(change, match):
    a, b, c = records()
    with pytest.raises(ValueError, match=match):
        join_originals([a | change], [b], [c])
