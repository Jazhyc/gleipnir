from pathlib import Path

import numpy as np
import pytest

from gleipnir.plotting import pareto_frontier_mask
from scripts.plot_tool_trajectory_ood_frontier import (
    DEFAULT_SOURCE,
    _display_label,
    _label_obstacles,
    _markdown_cells,
    load_frontier_registry,
    select_plot_points,
)


def test_pareto_frontier_keeps_nondominated_and_duplicate_points() -> None:
    mask = pareto_frontier_mask(
        [1.0, 2.0, 2.0, 3.0, 4.0],
        [0.5, 0.4, 0.6, 0.7, 0.7],
    )
    assert mask.tolist() == [True, False, True, True, False]

    duplicates = pareto_frontier_mask([1.0, 1.0], [0.5, 0.5])
    assert duplicates.tolist() == [True, True]


def test_pareto_frontier_rejects_invalid_coordinates() -> None:
    with pytest.raises(ValueError, match="positive"):
        pareto_frontier_mask([0.0], [0.5])
    with pytest.raises(ValueError, match="paired"):
        pareto_frontier_mask([1.0], [0.5, 0.6])
    with pytest.raises(ValueError, match="finite"):
        pareto_frontier_mask([1.0], [np.nan])


def test_markdown_cells_preserve_escaped_decision_pipe() -> None:
    cells = _markdown_cells(
        r"| Gleipnir | Model | Immediate `0\|1` logits | 1.0 | 0.8 | yes |"
    )
    assert cells == [
        "Gleipnir",
        "Model",
        "Immediate `0|1` logits",
        "1.0",
        "0.8",
        "yes",
    ]


def test_canonical_frontier_registry_parses_and_matches_nondominance() -> None:
    frame = load_frontier_registry(DEFAULT_SOURCE)

    assert len(frame) == 26
    assert frame["computed_frontier"].sum() == 11
    assert frame["computed_frontier"].equals(frame["declared_frontier"])
    kimi = frame.loc[frame["monitor"] == "Kimi K3 binary logits"].iloc[0]
    assert kimi["cost_per_1k"] == pytest.approx(25.1177)
    assert kimi["mean_ood_pauroc_at_20"] == pytest.approx(0.9084)


def test_plot_applies_documented_filters_without_changing_registry() -> None:
    frame = load_frontier_registry(DEFAULT_SOURCE)
    plotted = select_plot_points(frame)

    assert "Claude Opus 4.6 prompted" in set(frame["monitor"])
    assert "Claude Opus 4.6 prompted" not in set(plotted["monitor"])
    assert frame["mean_ood_pauroc_at_20"].min() < 0.60
    assert plotted["mean_ood_pauroc_at_20"].min() >= 0.60
    ours = plotted.loc[plotted["origin"] == "Gleipnir"]
    assert ours["interface"].str.contains("logits", case=False).all()
    assert "Qwen3.5-9B base (reasoned binary)" in set(frame["monitor"])
    assert "Qwen3.5-9B base (reasoned binary)" not in set(plotted["monitor"])
    assert "Qwen3.5-9B base (reasoned ordinal)" not in set(plotted["monitor"])


def test_display_labels_drop_redundant_base_suffix() -> None:
    assert _display_label("Qwen3.5-4B base") == "Qwen3.5-4B"
    assert _display_label("Qwen3.5-9B base (ordinal)") == "Qwen3.5-9B (ordinal)"
    assert _display_label("Kimi K3 binary logits") == "Kimi K3"
    assert _display_label("GPT-OSS-20B SFT") == "GPT-OSS-20B SFT"


def test_label_obstacles_sample_the_frontier_between_points() -> None:
    plotted = select_plot_points(load_frontier_registry(DEFAULT_SOURCE))
    obstacle_x, obstacle_y = _label_obstacles(plotted)

    assert obstacle_x.shape == obstacle_y.shape
    assert len(obstacle_x) > len(plotted)
    assert np.isfinite(obstacle_x).all()
    assert (obstacle_x > 0).all()


def test_registry_loader_rejects_a_stale_frontier_column(tmp_path: Path) -> None:
    registry = tmp_path / "frontier.md"
    source = DEFAULT_SOURCE.read_text(encoding="utf-8")
    registry.write_text(
        source.replace(
            "| **Gleipnir** | **Qwen3.5-4B base** |",
            "| **Gleipnir** | **Qwen3.5-4B base** |",
        ).replace("| **0.6175** | **yes** |", "| **0.6175** | **no** |", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Qwen3.5-4B base"):
        load_frontier_registry(registry)
