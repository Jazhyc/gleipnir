from pathlib import Path

import numpy as np
import pytest

from gleipnir.plotting import pareto_frontier_mask
from scripts.plot_tool_trajectory_ood_frontier import (
    DEFAULT_SOURCE,
    MANUAL_COMPARISON_LABELS,
    _display_label,
    _improvement_pairs,
    _label_obstacles,
    _markdown_cells,
    _paper_frontier,
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


def test_display_labels_drop_redundant_base_suffix() -> None:
    assert _display_label("Qwen3.5-4B base") == "Qwen3.5-4B"
    assert _display_label("Qwen3.5-9B base (ordinal)") == "Qwen3.5-9B (ordinal)"
    assert _display_label("Kimi K3 binary logits") == "Kimi K3"
    assert _display_label("Kimi K2.6 binary logits") == "Kimi K2.6"
    assert _display_label("Kimi-K2.6 SFT") == "Kimi-K2.6 SFT"
    assert _display_label("GPT-OSS-20B SFT") == "GPT-OSS-20B SFT"
    assert _display_label("Qwen3.5-27B base") == "Qwen3.5-27B"
    assert _display_label("Qwen3.5-4B SFT+RL") == "Qwen3.5-4B SFT+RL"
    assert _display_label("Qwen3.5-27B SFT+RL") == "Qwen3.5-27B SFT+RL"
    assert _display_label("Qwen3.5-4B Kimi-soft strict seed average") == "Gleipnir 4B"
    assert _display_label("Qwen3.5-9B Kimi-soft mixed") == "Gleipnir 9B"


def test_manual_comparison_labels_use_requested_sides_of_points() -> None:
    assert MANUAL_COMPARISON_LABELS["Kimi K3 binary logits"] == {
        "xytext": (0, -18),
        "ha": "center",
        "va": "top",
    }
    assert MANUAL_COMPARISON_LABELS["Kimi-K2.6 SFT"] == {
        "xytext": (0, -18),
        "ha": "center",
        "va": "top",
    }
    assert MANUAL_COMPARISON_LABELS["Qwen3.5-4B SFT+RL"]["xytext"] == (18, -14)
    # Qwen3.8-27B sits above Qwen3.5-27B at the shared 27B cost; its trained
    # point is up and left, so the base label moves right.
    assert MANUAL_COMPARISON_LABELS["Qwen3.5-27B base"]["xytext"] == (0, -18)
    assert MANUAL_COMPARISON_LABELS["Qwen3.8-27B base"]["xytext"] == (10, -4)
    assert MANUAL_COMPARISON_LABELS["Qwen3.8-27B Kimi-soft strict"]["xytext"] == (
        -10,
        6,
    )
    assert MANUAL_COMPARISON_LABELS["Qwen3.5-27B SFT+RL"] == {
        "xytext": (0, -36),
        "ha": "center",
        "va": "top",
    }
    assert MANUAL_COMPARISON_LABELS["Claude Sonnet 4.6 prompted"] == {
        "xytext": (0, -18),
        "ha": "center",
        "va": "top",
    }
    assert MANUAL_COMPARISON_LABELS["Gemini 3.1 Pro prompted"] == {
        "xytext": (0, 18),
        "ha": "center",
        "va": "bottom",
    }


def test_improvement_arrows_pair_each_gleipnir_method_with_its_base() -> None:
    plotted = select_plot_points(load_frontier_registry(DEFAULT_SOURCE))
    pairs = _improvement_pairs(plotted)

    assert [
        (str(base["monitor"]), str(method["monitor"])) for base, method in pairs
    ] == [
        ("Qwen3.5-4B base", "Qwen3.5-4B Kimi-soft strict seed average"),
        ("Qwen3.8-27B base", "Qwen3.8-27B Kimi-soft strict"),
    ]
    assert all(
        float(method["mean_ood_pauroc_at_20"]) > float(base["mean_ood_pauroc_at_20"])
        for base, method in pairs
    )


def test_paper_frontier_reconstructs_the_pre_gleipnir_low_cost_curve() -> None:
    plotted = select_plot_points(load_frontier_registry(DEFAULT_SOURCE))
    paper_frontier = _paper_frontier(plotted)

    assert paper_frontier.iloc[0]["monitor"] == "GPT-OSS-20B SFT"
    assert "Qwen3.5-4B SFT+RL" in set(paper_frontier["monitor"])
    assert "Qwen3.5-27B SFT+RL" in set(paper_frontier["monitor"])
    assert paper_frontier["cost_per_1k"].is_monotonic_increasing
    assert paper_frontier["mean_ood_pauroc_at_20"].is_monotonic_increasing


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
        source.replace("| **0.9508** | **yes** |", "| **0.9508** | **no** |", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Qwen3.8-27B Kimi-soft strict"):
        load_frontier_registry(registry)


def test_pricing_regimes_have_separate_frontiers() -> None:
    frame = load_frontier_registry(DEFAULT_SOURCE).set_index("monitor")
    committed = [
        "Qwen3.5-4B Kimi-soft strict seed average (committed GPUs)",
        "Qwen3.8-27B Kimi-soft strict (committed GPUs)",
    ]
    on_demand = [
        "Qwen3.5-4B Kimi-soft strict seed average",
        "Qwen3.8-27B Kimi-soft strict",
    ]
    decider = "pplx-decider v1.2 native readout"

    assert (frame.loc[committed, "pricing"] == "committed").all()
    assert (frame.loc[on_demand, "pricing"] == "on_demand").all()
    assert frame.loc[committed, "committed_frontier"].all()
    assert not frame.loc[committed, "on_demand_frontier"].any()
    assert frame.loc[on_demand + [decider], "on_demand_frontier"].all()
    assert not frame.loc[[decider], "committed_frontier"].any()


def test_main_figure_simplifies_while_full_keeps_every_eligible_row() -> None:
    frame = load_frontier_registry(DEFAULT_SOURCE)
    main = select_plot_points(frame).set_index("monitor")
    full = select_plot_points(frame, full=True).set_index("monitor")

    assert "d1 native probabilities" in full.index
    assert "d1 native probabilities" not in main.index
    assert "Kimi K2.6 binary logits" not in main.index
    assert {"Kimi K3 binary logits", "GPT-6 Luna binary logprobs"} <= set(main.index)
    assert main.loc["Qwen3.8-27B Kimi-soft strict", "category"] == "Gleipnir"
    assert main.loc["pplx-decider v1.2 native readout", "category"] == "Third-party"
    paper = main.loc[main["origin"].eq("Paper")]
    assert len(paper) < int(full["origin"].eq("Paper").sum())
