import pandas as pd
import pytest

from gleipnir.metrics import evaluate_binary_monitor, normalized_partial_auroc


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
    assert result["macro"]["pauroc_at_20"] == pytest.approx(0.5)
    assert result["macro"]["balanced_accuracy"] == pytest.approx(0.5)
    assert result["diagnostics"] == {"unique_scores": 4, "tied_rows": 0}


def test_metrics_reject_out_of_range_scores() -> None:
    frame = pd.DataFrame({"dataset": ["a"], "label": [0], "score": [1.1]})
    with pytest.raises(ValueError, match="lie in"):
        evaluate_binary_monitor(frame)


def test_partial_auroc_uses_raw_normalized_paper_definition() -> None:
    labels = pd.Series([0, 1]).to_numpy()
    tied_scores = pd.Series([0.5, 0.5]).to_numpy()
    perfect_scores = pd.Series([0.0, 1.0]).to_numpy()

    assert normalized_partial_auroc(labels, tied_scores) == pytest.approx(0.1)
    assert normalized_partial_auroc(labels, perfect_scores) == pytest.approx(1.0)


def test_partial_auroc_clips_and_interpolates_at_fpr_budget() -> None:
    labels = pd.Series([1, 0, 1, 0, 0]).to_numpy()
    scores = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5]).to_numpy()

    # The ROC rises to TPR=0.5 at FPR=0, then stays flat through FPR=1/3.
    assert normalized_partial_auroc(labels, scores) == pytest.approx(0.5)
