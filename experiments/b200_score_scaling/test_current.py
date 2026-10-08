"""Current-recipe scaling must retain one cohort and complete finite controls."""

import copy
import json
from pathlib import Path

import pytest

from experiments.b200_score_scaling.current import validate_contract
from experiments.b200_score_scaling.current_summary import aggregate


def fixture():
    settings = json.loads((Path(__file__).parent / "current_config.json").read_text())
    rows = [
        {"id": str(i), "prompt_sha256": str(i), "prompt_tokens": 4096}
        for i in range(320)
    ]
    rows[-1]["prompt_tokens"] -= 139
    values = [r | {"score": 0.5, "margin": 0.0} for r in rows]
    return settings, rows, [copy.deepcopy(values) for _ in range(6)]


def test_full_cohort_is_identical_at_every_level():
    settings, rows, controls = fixture()
    validate_contract(settings, rows, controls)
    with pytest.raises(ValueError, match="workload/repeat"):
        validate_contract(settings, rows[:64], controls)


def test_reference_order_and_token_counts_are_checked():
    settings, rows, controls = fixture()
    controls[-1][0]["prompt_tokens"] += 1
    with pytest.raises(ValueError, match="reference identity"):
        validate_contract(settings, rows, controls)
    settings, rows, controls = fixture()
    controls[-1].reverse()
    with pytest.raises(ValueError, match="reference identity"):
        validate_contract(settings, rows, controls)


def test_incomplete_or_nonfinite_controls_are_rejected():
    settings, rows, controls = fixture()
    with pytest.raises(ValueError, match="workload/repeat"):
        validate_contract(settings, rows, controls[:-1])
    controls[0][0]["score"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        validate_contract(settings, rows, controls)


def test_promotion_and_extra_client_levels_are_rejected():
    settings, rows, controls = fixture()
    settings["promote"] = True
    with pytest.raises(ValueError, match="workload/repeat"):
        validate_contract(settings, rows, controls)
    settings["promote"] = False
    settings["concurrencies"].append(256)
    with pytest.raises(ValueError, match="workload/repeat"):
        validate_contract(settings, rows, controls)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "cropped"])
def test_summary_rejects_incomplete_or_mixed_repeats(mutation):
    trials = [
        {
            "concurrency": 1,
            "repeat": i,
            "prompt_tokens_per_second": 100.0 + i,
            "requests_per_second": 1.0,
            "latency": {
                "rows": 320,
                "p50_seconds": 1.0,
                "p95_seconds": 2.0,
                "p99_seconds": 3.0,
            },
        }
        for i in range(3)
    ]
    if mutation == "missing":
        trials.pop()
    elif mutation == "duplicate":
        trials[-1]["repeat"] = 0
    else:
        trials[-1]["latency"]["rows"] = 64
    with pytest.raises(ValueError, match="repeat/cohort"):
        aggregate({"status": "complete", "trials": trials}, (1,))
