from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from scripts.plot_tool_trajectory_ood_scaling import (
    DEFAULT_SOURCE,
    MATCHED_ROW_SCHEDULE,
    load_paper_scaling_reference,
    load_scaling_results,
    plot_scaling,
)


def test_canonical_scaling_results_select_only_matched_9b_curve() -> None:
    frame = load_scaling_results(DEFAULT_SOURCE)

    assert frame["rows"].tolist() == MATCHED_ROW_SCHEDULE
    assert set(frame["backbone"]) == {"Qwen3.5-9B"}
    assert set(frame["training_data"]) == {"matched"}
    assert frame.loc[frame["mean_ood_pauroc_at_20"].idxmax(), "rows"] == 4_008


def test_canonical_paper_reference_contains_digitized_4b_and_27b_curves() -> None:
    frame = load_paper_scaling_reference(DEFAULT_SOURCE)

    assert set(frame["backbone"]) == {"Qwen3.5-4B", "Qwen3.5-27B"}
    for _, group in frame.groupby("backbone"):
        assert group["rows"].tolist() == MATCHED_ROW_SCHEDULE
    qwen_4b = frame.loc[frame["backbone"].eq("Qwen3.5-4B")]
    qwen_27b = frame.loc[frame["backbone"].eq("Qwen3.5-27B")]
    assert qwen_4b["mean_ood_pauroc_at_20"].tolist() == pytest.approx(
        [0.628, 0.675, 0.666, 0.708, 0.697, 0.700]
    )
    assert qwen_27b["mean_ood_pauroc_at_20"].tolist() == pytest.approx(
        [0.766, 0.783, 0.779, 0.772, 0.785, 0.807]
    )


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


def test_paper_scaling_loader_fails_if_schedule_drifts(tmp_path: Path) -> None:
    source = tmp_path / "scaling.md"
    source.write_text(
        DEFAULT_SOURCE.read_text(encoding="utf-8").replace(
            "| Qwen3.5-27B | 504 |",
            "| Qwen3.5-27B | 505 |",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="paper Qwen3.5-27B row schedule changed"):
        load_paper_scaling_reference(source)


def test_scaling_plot_compares_only_pauroc_curves() -> None:
    figure = plot_scaling(
        load_scaling_results(DEFAULT_SOURCE),
        load_paper_scaling_reference(DEFAULT_SOURCE),
    )
    try:
        axis = figure.axes[0]
        assert len(axis.lines) == 3
        assert axis.get_xscale() == "log"
        labels = {line.get_label() for line in axis.lines}
        assert labels == {
            "Ours · 9B soft logits",
            "Sinha et al. · 4B rationale SFT",
            "Sinha et al. · 27B rationale SFT",
        }
        assert all("AUROC" not in label for label in labels)
        annotations = {text.get_text() for text in axis.texts}
        assert "Our best: 83.7%\n4,008 rows" in annotations
        assert "Ours · 9B soft logits" in annotations
        assert "Sinha et al. · 4B rationale SFT" in annotations
        assert "Sinha et al. · 27B rationale SFT" in annotations
    finally:
        plt.close(figure)
