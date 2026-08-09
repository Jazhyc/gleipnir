import pandas as pd
import pytest

from gleipnir.metrics import evaluate_binary_monitor


def test_metrics_are_macro_averaged_across_groups() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["a", "a", "b", "b"],
            "label": [0, 1, 0, 1],
            "score": [0.1, 0.9, 0.8, 0.2],
        }
    )
    result = evaluate_binary_monitor(frame)
    assert result["macro"]["auroc"] == pytest.approx(0.5)
    assert result["macro"]["balanced_accuracy"] == pytest.approx(0.5)
    assert result["diagnostics"] == {"unique_scores": 4, "tied_rows": 0}


def test_metrics_reject_out_of_range_scores() -> None:
    frame = pd.DataFrame({"dataset": ["a"], "label": [0], "score": [1.1]})
    with pytest.raises(ValueError, match="lie in"):
        evaluate_binary_monitor(frame)
