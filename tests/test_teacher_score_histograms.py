import json

import pytest

from scripts.plot_teacher_score_histograms import load_scores


def test_paired_scores_preserve_endpoints(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps({"id": "a", "qwen_score": 0, "kimi_score": 1}))
    qwen, kimi = load_scores(path)
    assert qwen.tolist() == [0] and kimi.tolist() == [1]


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_invalid_probability_rejected(tmp_path, value):
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps({"id": "a", "qwen_score": value, "kimi_score": 0.5}))
    with pytest.raises(ValueError, match="finite probabilities"):
        load_scores(path)


def test_duplicate_pairs_rejected(tmp_path):
    path = tmp_path / "scores.jsonl"
    row = json.dumps({"id": "a", "qwen_score": 0, "kimi_score": 1})
    path.write_text(row + "\n" + row)
    with pytest.raises(ValueError, match="unique paired"):
        load_scores(path)
