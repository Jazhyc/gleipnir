"""Activation-only ranking, parent closure and matched-control invariants."""

import pytest

from gleipnir.data.activation_filter import activation_exclusions, matched_exclusions


def rows() -> list[dict]:
    return [
        {
            "index": str(i),
            "label": i % 2,
            "source": "s",
            "trajectory_sha256": str(i),
            "delta_z20": float(i),
            "soft_target": 0.9,
            "prompt_tokens": 100 + i,
        }
        for i in range(40)
    ]


def test_activation_filter_ignores_teacher_and_closes_content() -> None:
    data = rows()
    selected = activation_exclusions(data, 0.2)
    assert selected == {"32", "34", "36", "38"}
    changed = [r | {"soft_target": 0, "firewall_score": 0} for r in data]
    assert activation_exclusions(changed, 0.2) == selected
    changed[0]["trajectory_sha256"] = "38"
    assert activation_exclusions(changed, 0.2) == selected | {"0"}
    changed[1]["trajectory_sha256"] = "38"
    with pytest.raises(ValueError, match="harmful"):
        activation_exclusions(changed, 0.2)


def test_random_control_is_reproducible_and_matches_strata() -> None:
    data = rows()
    selected = activation_exclusions(data, 0.2)
    injected = {str(i) for i in range(0, 40, 4)}
    a, receipt = matched_exclusions(data, selected, injected, 17)
    b, repeated = matched_exclusions(list(reversed(data)), selected, injected, 17)
    assert a == b and receipt == repeated and len(a) == len(selected)
    assert all(next(r for r in data if r["index"] == i)["label"] == 0 for i in a)
    assert sum(i in injected for i in a) == sum(i in injected for i in selected)
