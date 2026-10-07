import pytest

from gleipnir.calibration import binary_calibration


def test_perfect_endpoints_and_empty_bins():
    s = binary_calibration([0, 1], [0, 1])
    assert s["ece"] == s["brier"] == 0
    assert sum(b["n"] for b in s["bins"]) == 2
    assert s["log_loss"] < 1e-14


def test_known_gaps_and_boundaries():
    s = binary_calibration([0, 1, 0, 1], [0.1, 0.1, 0.9, 0.9])
    assert s["ece"] == pytest.approx(0.4)
    assert s["brier"] == pytest.approx(0.41)
    assert s["bins"][1]["n"] == s["bins"][9]["n"] == 2
    assert s["confidence_minus_accuracy"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "y,p", [([], []), ([0], [2]), ([2], [0.1]), ([0], [float("nan")]), ([0], [0, 1])]
)
def test_invalid(y, p):
    with pytest.raises(ValueError):
        binary_calibration(y, p)
