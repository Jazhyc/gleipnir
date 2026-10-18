"""Length bins are assigned by rank inside each stratum."""

import pandas as pd

from experiments.shade_length_audit.analyze import within_stratum_bins


def test_bins_balance_each_stratum() -> None:
    frame = pd.DataFrame(
        {
            "task": ["a"] * 8 + ["b"] * 4,
            "tokens": [80, 10, 70, 20, 60, 30, 50, 40, 9000, 1000, 3000, 2000],
        }
    )
    bins = within_stratum_bins(frame, ["task"], bins=4)
    assert bins[frame["task"] == "a"].value_counts().sort_index().tolist() == [2] * 4
    assert bins[frame["task"] == "b"].tolist() == [3, 0, 2, 1]
    # Bins follow rank inside the stratum, not absolute token counts.
    assert bins.iloc[7] == 1 and bins.iloc[8] == 3


def test_singleton_stratum_is_top_bin() -> None:
    frame = pd.DataFrame({"task": ["a"], "tokens": [5]})
    assert within_stratum_bins(frame, ["task"], bins=4).tolist() == [3]
