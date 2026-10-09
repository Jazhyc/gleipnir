"""Control construction must preserve parent text and paired diagnostic meaning."""

from types import SimpleNamespace

import pytest

from experiments.b200_judge_direction_signal.analyze import detection
from experiments.b200_judge_direction_signal.prepare import insertion, match_padding


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(
            ids=list(range(len(text))), offsets=[(i, i + 1) for i in range(len(text))]
        )


def test_single_insertion_recovers_parent_with_repeated_boundary_text():
    clean = "<candidate_A>repeat repeat</candidate_A>"
    attacked = clean.replace("</candidate_A>", " repeated text</candidate_A>")
    start, suffix = insertion(clean, attacked)
    assert attacked[:start] + attacked[start + len(suffix) :] == clean
    assert clean[:start] + suffix + clean[start:] == attacked


@pytest.mark.parametrize("attacked", ["abcYef", "XabcdefY", "abcdef"])
def test_replacements_multiple_changes_and_empty_insertions_rejected(attacked):
    with pytest.raises(ValueError, match="insertion"):
        insertion("abcdef", attacked)


def test_control_matching_counts_complete_prompt_and_preserves_location():
    clean = "abcdef"
    start = 3
    target = 21
    suffix, count = match_padding(
        CharacterTokenizer(), clean, start, target, " ordinary text."
    )
    assert abs(count - target) <= 1
    assert len(clean[:start] + suffix + clean[start:]) == count
    assert (clean[:start] + suffix + clean[start:]).replace(suffix, "", 1) == clean


def test_fixed_direction_sign_and_matched_parent_weighting():
    clean = {"a": {"z20": 10.0}, "b": {"z20": -10.0}}
    injected = [
        {"clean_id": "a", "z20": 9.0},
        {"clean_id": "a", "z20": 8.0},
        {"clean_id": "b", "z20": -11.0},
    ]
    result = detection(injected, clean)
    assert result["paired_shift"]["mean"] == pytest.approx(-4 / 3)
    assert result["paired_shift"]["fraction_positive"] == 0
    assert result["auroc"] < 0.5
    assert result["unique_clean_parents"] == 2
