from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from scripts.plot_tool_trajectory_ood_scaling import (
    DEFAULT_SOURCE,
    MATCHED_ROW_SCHEDULE,
    load_scaling_results,
    plot_scaling,
)


def test_canonical_scaling_results_select_only_matched_9b_curve() -> None:
    frame = load_scaling_results(DEFAULT_SOURCE)

    assert frame["rows"].tolist() == MATCHED_ROW_SCHEDULE
    assert set(frame["backbone"]) == {"Qwen3.5-9B"}
    assert set(frame["training_data"]) == {"matched"}
    assert frame.loc[frame["mean_ood_auroc"].idxmax(), "rows"] == 996
    assert frame.loc[frame["mean_ood_pauroc_at_20"].idxmax(), "rows"] == 4_008


def test_scaling_loader_fails_if_canonical_schedule_drifts(tmp_path: Path) -> None:
    source = tmp_path / "scaling.md"
    source.write_text(
        DEFAULT_SOURCE.read_text(encoding="utf-8").replace(
            "| Qwen3.5-9B | matched | 504 |",
            "| Qwen3.5-9B | matched | 505 |",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="row schedule changed"):
        load_scaling_results(source)


def test_scaling_plot_has_both_metrics_and_peak_annotations() -> None:
    figure = plot_scaling(load_scaling_results(DEFAULT_SOURCE))
    try:
        axis = figure.axes[0]
        assert len(axis.lines) == 2
        assert axis.get_xscale() == "log"
        labels = {line.get_label() for line in axis.lines}
        assert labels == {"Mean-OOD pAUROC@20", "Mean-OOD AUROC"}
        annotations = {text.get_text() for text in axis.texts}
        assert "Best AUROC: 93.5%\n996 rows" in annotations
        assert "Best pAUROC@20: 83.7%\n4,008 rows" in annotations
    finally:
        plt.close(figure)
