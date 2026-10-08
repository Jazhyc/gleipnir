"""A finite diagnostic preserves agreement failure and cannot waive output guards."""

import pytest

from experiments.b200_vllm031.run import may_benchmark


def failed_canary():
    return {
        "finite": True,
        "adapter_effect": 0.8,
        "correlation": 0.997,
        "mean_absolute_difference": 0.012,
        "passed": False,
    }


def test_failed_agreement_requires_explicit_diagnostic():
    canary = failed_canary()
    assert not may_benchmark(canary, False)
    assert may_benchmark(canary, True)
    assert canary["passed"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("finite", False),
        ("adapter_effect", 0.0),
        ("correlation", float("nan")),
        ("mean_absolute_difference", float("inf")),
    ],
)
def test_diagnostic_cannot_waive_invalid_output(field, value):
    canary = {**failed_canary(), field: value}
    assert not may_benchmark(canary, True)
