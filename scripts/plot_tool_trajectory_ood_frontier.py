"""Plot the canonical tool-trajectory OOD cost--performance frontier."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import textwrap
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gleipnir-matplotlib")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter

from gleipnir.plotting import (
    ORIGIN_PALETTE,
    pareto_frontier_mask,
    save_figure,
    set_plot_style,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPOSITORY_ROOT / "docs/research/tool_trajectory_ood_frontier.md"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "figures/tool_trajectory_ood_frontier.png"
DEFAULT_VECTOR_OUTPUT = REPOSITORY_ROOT / "figures/tool_trajectory_ood_frontier.svg"
FULL_OUTPUTS = [
    REPOSITORY_ROOT / "figures/tool_trajectory_ood_frontier_full.png",
    REPOSITORY_ROOT / "figures/tool_trajectory_ood_frontier_full.svg",
]
EXCLUDED_MONITORS = {
    # Ties Gemini 3.1 Pro (0.942 vs 0.940) but would stretch the x-axis past $80.
    "Claude Opus 4.6 prompted",
    "Qwen3.5-9B base (reasoned binary)",
    # Superseded by Gleipnir 4B v0.3 and no longer optimized (user, 2026-10-20);
    # the registry keeps its row.
    "Qwen3.5-9B Kimi-soft mixed",
}
# Omitted from the main figure only (user, 2026-10-20); the full appendix
# figure keeps them. Less prominent decision models (reported in the final
# report's table instead), early base-model references without an improvement
# arrow, and Kimi K2.6 (the teacher is K3).
MAIN_FIGURE_OMISSIONS = {
    "Microsoft-Decision-1 native probabilities",
    "d1 native probabilities",
    "Mercury Decide native probabilities",
    "d1-ultra native probabilities",
    "Qwen3.5-9B base",
    "Qwen3.5-27B base",
    "Qwen3.5-35B-A3B base (mixed providers)",
    "Kimi K2.6 binary logits",
}
CATEGORY_PALETTE = {**ORIGIN_PALETTE, "Other": "#B279A2"}
CATEGORY_MARKERS = {"Paper": "o", "Gleipnir": "D", "Other": "s"}
CATEGORY_LABELS = {
    "Paper": "Reasoning (Sinha et al.)",
    "Gleipnir": "Gleipnir (ours)",
    "Other": "Decision",
}
LEGEND_ORDER = ("Paper", "Other", "Gleipnir")
GLEIPNIR_METHODS = {
    "Qwen3.5-4B Kimi-soft strict seed average": "Gleipnir 4B",
    "Qwen3.5-9B Kimi-soft mixed": "Gleipnir 9B",
    "Qwen3.8-27B Kimi-soft strict": "Gleipnir 27B",
    "Qwen3.5-4B Kimi-soft strict seed average (committed GPUs)": "Gleipnir 4B*",
    "Qwen3.8-27B Kimi-soft strict (committed GPUs)": "Gleipnir 27B*",
}
# Self-hosted rows are priced at on-demand or committed-use GPU rates. Each
# regime gets its own frontier against the list-priced rows.
ON_DEMAND_MARKER = "on-demand B200 break-even"
COMMITTED_MARKER = "committed-use B200 break-even"
BASE_TO_METHOD = {
    "Qwen3.5-4B base": "Qwen3.5-4B Kimi-soft strict seed average",
    "Qwen3.5-9B base": "Qwen3.5-9B Kimi-soft mixed",
    "Qwen3.8-27B base": "Qwen3.8-27B Kimi-soft strict",
}
# Labelled paper points. Gemini 3.1 Pro marks the top of the paper frontier.
# The main figure keeps these points on the paper frontier but drops their
# labels: Kimi-K2.6 SFT's logit-based counterpart is not shown there, and the
# SFT+RL labels served the interface-bottleneck comparison, which the figure no
# longer emphasizes over decision models (user, 2026-10-20).
MAIN_FIGURE_UNLABELLED = {"Kimi-K2.6 SFT", "Qwen3.5-4B SFT+RL", "Qwen3.5-27B SFT+RL"}
PAPER_COMPARISON_MONITORS = {
    "Gemini 3.1 Pro prompted",
    "Kimi-K2.6 SFT",
    "Qwen3.5-4B SFT+RL",
    "Qwen3.5-27B SFT+RL",
}
MANUAL_COMPARISON_LABELS = {
    "Jev 1.13 native probabilities": {
        "xytext": (0, -14),
        "ha": "center",
        "va": "top",
    },
    "GPT-6 Luna binary logprobs": {
        "xytext": (-5, 8),
        "ha": "right",
        "va": "bottom",
    },
    # Below-right, clear of the y-axis tick labels.
    "Qwen3.5-4B Kimi-soft strict seed average (committed GPUs)": {
        "xytext": (-2, -14),
        "ha": "left",
        "va": "top",
    },
    # Above its point, between the Decider and the on-demand 27B.
    "Qwen3.8-27B Kimi-soft strict (committed GPUs)": {
        "xytext": (-8, 12),
        "ha": "right",
        "va": "bottom",
    },
    # Below-right, clear of both Gleipnir frontiers. Mercury Decide (full
    # figure only) goes further below so the two labels stay apart.
    "pplx-decider v1.2 native readout": {
        "xytext": (12, -12),
        "ha": "left",
        "va": "top",
    },
    "Mercury Decide native probabilities": {
        "xytext": (12, -21),
        "ha": "left",
        "va": "top",
    },
    "d1-ultra native probabilities": {
        "xytext": (10, 0),
        "ha": "left",
        "va": "center",
    },
    # Left of its point, clear of the improvement arrow from the base 4B.
    "d1 native probabilities": {
        "xytext": (-10, 0),
        "ha": "right",
        "va": "center",
    },
    # Above its point, clear of the improvement arrow from the base 4B.
    "Qwen3.5-4B Kimi-soft strict seed average": {
        "xytext": (0, 14),
        "ha": "center",
        "va": "bottom",
    },
    "Qwen3.5-9B Kimi-soft mixed": {
        "xytext": (0, -10),
        "ha": "center",
        "va": "top",
    },
    "Qwen3.5-35B-A3B base (mixed providers)": {
        "xytext": (12, -16),
        "ha": "left",
        "va": "top",
    },
    "Qwen3.5-9B base": {"xytext": (12, -3), "ha": "left", "va": "top"},
    "Kimi K3 binary logits": {
        "xytext": (0, -18),
        "ha": "center",
        "va": "top",
    },
    # Below its point: the frontier now passes just above K2.6 logits.
    "Kimi K2.6 binary logits": {
        "xytext": (0, -18),
        "ha": "center",
        "va": "top",
    },
    "Kimi-K2.6 SFT": {
        "xytext": (0, -18),
        "ha": "center",
        "va": "top",
    },
    "Qwen3.5-4B SFT+RL": {"xytext": (18, -14), "ha": "left", "va": "top"},
    "Qwen3.5-27B base": {"xytext": (0, -18), "ha": "center", "va": "top"},
    # The trained 27B sits directly above its base at the same cost.
    "Qwen3.8-27B base": {"xytext": (10, -4), "ha": "left", "va": "top"},
    "Qwen3.8-27B Kimi-soft strict": {
        "xytext": (-10, 6),
        "ha": "right",
        "va": "bottom",
    },
    "Qwen3.5-27B SFT+RL": {"xytext": (0, -36), "ha": "center", "va": "top"},
    "Claude Sonnet 4.6 prompted": {
        "xytext": (0, -18),
        "ha": "center",
        "va": "top",
    },
    # Above its point, clear of the cost callout arrow; it sits at the right
    # edge of the axes.
    "Gemini 3.1 Pro prompted": {
        "xytext": (4, 12),
        "ha": "right",
        "va": "bottom",
    },
}
# Cost-saving callout: an arrow from the strongest paper monitor to the
# on-demand Gleipnir 27B, labelled with the cost ratio (user, 2026-10-20).
COST_CALLOUT = ("Gemini 3.1 Pro prompted", "Qwen3.8-27B Kimi-soft strict")
# Fraction along each base-to-Gleipnir arrow where its label sits.
DISTILLATION_LABEL_POSITION = 0.6
MIN_PLOTTED_PERFORMANCE = 0.60
EXPECTED_COLUMNS = [
    "Origin",
    "Monitor",
    "Evaluation interface",
    "Uncached inference cost (USD / 1,000 evaluations)",
    "Mean-OOD pAUROC@20",
    "Frontier",
]


def _markdown_cells(line: str) -> list[str]:
    cells = re.split(r"(?<!\\)\|", line.strip())
    if cells and not cells[0]:
        cells = cells[1:]
    if cells and not cells[-1]:
        cells = cells[:-1]
    return [cell.strip().replace(r"\|", "|") for cell in cells]


def _plain_text(value: str) -> str:
    return value.replace("**", "").replace("`", "").strip()


def load_frontier_registry(path: str | Path) -> pd.DataFrame:
    """Load and validate the canonical Markdown reference table."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if _markdown_cells(line) == EXPECTED_COLUMNS
        ),
        None,
    )
    if header_index is None:
        raise ValueError(f"reference table header not found in {path}")

    rows: list[list[str]] = []
    for line in lines[header_index + 2 :]:
        if not line.strip().startswith("|"):
            break
        cells = _markdown_cells(line)
        if len(cells) != len(EXPECTED_COLUMNS):
            raise ValueError(f"malformed reference-table row: {line}")
        rows.append([_plain_text(cell) for cell in cells])
    if not rows:
        raise ValueError(f"reference table has no data rows in {path}")

    frame = pd.DataFrame(rows, columns=EXPECTED_COLUMNS).rename(
        columns={
            EXPECTED_COLUMNS[0]: "origin",
            EXPECTED_COLUMNS[1]: "monitor",
            EXPECTED_COLUMNS[2]: "interface",
            EXPECTED_COLUMNS[3]: "cost_per_1k",
            EXPECTED_COLUMNS[4]: "mean_ood_pauroc_at_20",
            EXPECTED_COLUMNS[5]: "declared_frontier",
        }
    )
    frame["cost_per_1k"] = pd.to_numeric(frame["cost_per_1k"])
    frame["mean_ood_pauroc_at_20"] = pd.to_numeric(frame["mean_ood_pauroc_at_20"])
    declared = frame["declared_frontier"].str.lower()
    if not declared.isin({"yes", "no"}).all():
        raise ValueError("Frontier cells must be either 'yes' or 'no'")
    frame["declared_frontier"] = declared.eq("yes")
    frame["pricing"] = np.select(
        [
            frame["interface"].str.contains(ON_DEMAND_MARKER, regex=False),
            frame["interface"].str.contains(COMMITTED_MARKER, regex=False),
        ],
        ["on_demand", "committed"],
        default="list",
    )
    frame["on_demand_frontier"] = _regime_frontier(frame, "committed")
    frame["committed_frontier"] = _regime_frontier(frame, "on_demand")
    # A row's Frontier cell is its status within its own pricing regime;
    # list-priced rows report the on-demand regime.
    frame["computed_frontier"] = frame["on_demand_frontier"].where(
        frame["pricing"].ne("committed"), frame["committed_frontier"]
    )
    mismatches = frame.loc[
        frame["declared_frontier"] != frame["computed_frontier"], "monitor"
    ].tolist()
    if mismatches:
        raise ValueError(
            "declared frontier disagrees with recomputed nondominance for: "
            + ", ".join(mismatches)
        )
    return frame


def _regime_frontier(frame: pd.DataFrame, excluded_pricing: str) -> pd.Series:
    """Nondominance among rows outside one self-hosted pricing regime."""
    rows = frame["pricing"].ne(excluded_pricing)
    mask = pd.Series(False, index=frame.index)
    mask[rows] = pareto_frontier_mask(
        frame.loc[rows, "cost_per_1k"].to_numpy(),
        frame.loc[rows, "mean_ood_pauroc_at_20"].to_numpy(),
    )
    return mask


def _display_label(monitor: str) -> str:
    replacements = {
        "Kimi K2.6 binary logits": "Kimi K2.6",
        "Kimi K3 binary logits": "Kimi K3",
        "GPT-6 Luna binary logprobs": "GPT-6 Luna",
        "Jev 1.13 native probabilities": "Jev 1.13",
        "pplx-decider v1.2 native readout": "Decider v1.2",
        "Mercury Decide native probabilities": "Mercury Decide",
        "d1-ultra native probabilities": "d1-ultra",
        "d1 native probabilities": "d1",
        "Microsoft-Decision-1 native probabilities": "Microsoft-Decision-1",
        "Claude Sonnet 4.6 prompted": "Claude Sonnet 4.6",
        "Gemini 3.1 Pro prompted": "Gemini 3.1 Pro",
        "Claude Opus 4.6 prompted": "Claude Opus 4.6",
        "Qwen3.5-27B base": "Qwen3.5-27B",
        "Qwen3.8-27B base": "Qwen3.8-27B",
        "Qwen3.5-35B-A3B base (mixed providers)": "Qwen3.5-35B-A3B",
        **GLEIPNIR_METHODS,
    }
    display = replacements.get(monitor, monitor)
    display = re.sub(r"\s+base(?=$|\s+\()", "", display)
    return "\n".join(textwrap.fill(line, width=22) for line in display.split("\n"))


def select_plot_points(frame: pd.DataFrame, *, full: bool = False) -> pd.DataFrame:
    """Apply documented presentation-only omissions to registry points.

    The main figure also drops ``MAIN_FIGURE_OMISSIONS`` and paper points that
    are neither on the paper frontier nor labelled comparisons.
    """
    is_ours = frame["origin"].eq("Gleipnir")
    is_binary_score_interface = frame["interface"].str.contains(
        "logits|logprobs|native binary probability", case=False, na=False
    )
    included = (
        ~frame["monitor"].isin(EXCLUDED_MONITORS)
        & frame["mean_ood_pauroc_at_20"].ge(MIN_PLOTTED_PERFORMANCE)
        & (~is_ours | is_binary_score_interface)
    )
    if not full:
        paper = frame["origin"].eq("Paper") & included
        paper_frontier = pd.Series(False, index=frame.index)
        paper_frontier[paper] = pareto_frontier_mask(
            frame.loc[paper, "cost_per_1k"].to_numpy(),
            frame.loc[paper, "mean_ood_pauroc_at_20"].to_numpy(),
        )
        included &= ~frame["monitor"].isin(MAIN_FIGURE_OMISSIONS) & (
            ~paper | paper_frontier | frame["monitor"].isin(PAPER_COMPARISON_MONITORS)
        )
    selected = frame.loc[included].copy()
    selected["category"] = np.select(
        [
            selected["origin"].eq("Paper"),
            selected["monitor"].isin(GLEIPNIR_METHODS),
        ],
        ["Paper", "Gleipnir"],
        default="Other",
    )
    return selected


def _label_obstacles(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Sample points and frontier segments for label collision avoidance."""
    x_parts = [frame["cost_per_1k"].to_numpy()]
    y_parts = [frame["mean_ood_pauroc_at_20"].to_numpy()]
    segments = []
    for column in ("on_demand_frontier", "committed_frontier"):
        frontier = frame.loc[frame[column]].sort_values("cost_per_1k")
        segments += zip(
            frontier.iloc[:-1].iterrows(), frontier.iloc[1:].iterrows(), strict=True
        )
    for (_, start), (_, end) in segments:
        fractions = np.linspace(0.1, 0.9, 9)
        start_cost = float(start["cost_per_1k"])
        end_cost = float(end["cost_per_1k"])
        x_parts.append(
            np.exp(
                np.log(start_cost) + fractions * (np.log(end_cost) - np.log(start_cost))
            )
        )
        start_score = float(start["mean_ood_pauroc_at_20"])
        end_score = float(end["mean_ood_pauroc_at_20"])
        y_parts.append(start_score + fractions * (end_score - start_score))
    return np.concatenate(x_parts), np.concatenate(y_parts)


def _improvement_pairs(frame: pd.DataFrame) -> list[tuple[pd.Series, pd.Series]]:
    """Return available matched base-to-Gleipnir point pairs."""
    pairs = []
    for base_monitor, method_monitor in BASE_TO_METHOD.items():
        base = frame.loc[frame["monitor"].eq(base_monitor)]
        method = frame.loc[frame["monitor"].eq(method_monitor)]
        if len(base) == 1 and len(method) == 1:
            pairs.append((base.iloc[0], method.iloc[0]))
    return pairs


def _paper_frontier(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute the historical frontier using only visible paper points."""
    paper = frame.loc[frame["origin"].eq("Paper")].copy()
    paper["paper_frontier"] = pareto_frontier_mask(
        paper["cost_per_1k"].to_numpy(),
        paper["mean_ood_pauroc_at_20"].to_numpy(),
    )
    return paper.loc[paper["paper_frontier"]].sort_values("cost_per_1k")


def _draw_cost_callout(axis: plt.Axes, frame: pd.DataFrame) -> float | None:
    """Arrow from the reference monitor to Gleipnir 27B; returns the cost ratio."""
    source = frame.loc[frame["monitor"].eq(COST_CALLOUT[0])]
    target = frame.loc[frame["monitor"].eq(COST_CALLOUT[1])]
    if len(source) != 1 or len(target) != 1:
        return None
    source, target = source.iloc[0], target.iloc[0]
    start = (source["cost_per_1k"], source["mean_ood_pauroc_at_20"])
    end = (target["cost_per_1k"], target["mean_ood_pauroc_at_20"])
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#374151",
            "linewidth": 1.8,
            "mutation_scale": 16,
            "shrinkA": 9,
            "shrinkB": 11,
            "connectionstyle": "arc3,rad=0.04",
        },
        zorder=2,
    )
    ratio = source["cost_per_1k"] / target["cost_per_1k"]
    gain = 100 * (target["mean_ood_pauroc_at_20"] - source["mean_ood_pauroc_at_20"])
    axis.text(
        np.sqrt(start[0] * end[0]),
        (start[1] + end[1]) / 2 - 0.004,
        f"{ratio:.0f}× cheaper, {gain:+.1f} pp pAUROC",
        fontsize=13,
        fontweight="bold",
        color="#111827",
        ha="center",
        va="top",
        zorder=4,
    )
    return ratio


def _one_significant_figure(value: float) -> float:
    """Round to one significant figure for headline text (188 -> 200)."""
    return round(value, -int(np.floor(np.log10(abs(value)))))


def plot_frontier(frame: pd.DataFrame, *, full: bool = False) -> plt.Figure:
    """Build the publication-ready OOD frontier plot."""
    frame = select_plot_points(frame, full=full)
    set_plot_style()
    figure, axis = plt.subplots(figsize=(14.2, 8.5))
    axis.set_xscale("log")
    axis.set_xticks([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 80.0])
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_xlim(
        frame["cost_per_1k"].min() / 1.35,
        max(80.0, frame["cost_per_1k"].max() * 1.06),
    )
    axis.set_ylim(MIN_PLOTTED_PERFORMANCE - 0.02, 0.99)

    frontier = frame.loc[frame["on_demand_frontier"]].sort_values("cost_per_1k")
    committed_frontier = frame.loc[frame["committed_frontier"]].sort_values(
        "cost_per_1k"
    )
    paper_frontier = _paper_frontier(frame)
    axis.plot(
        paper_frontier["cost_per_1k"],
        paper_frontier["mean_ood_pauroc_at_20"],
        color=ORIGIN_PALETTE["Paper"],
        linestyle=(0, (1.5, 2.4)),
        linewidth=1.8,
        alpha=0.72,
        zorder=0,
    )
    axis.plot(
        frontier["cost_per_1k"],
        frontier["mean_ood_pauroc_at_20"],
        color="#374151",
        linewidth=2.3,
        alpha=0.82,
        zorder=1,
    )
    axis.plot(
        committed_frontier["cost_per_1k"],
        committed_frontier["mean_ood_pauroc_at_20"],
        color="#374151",
        linestyle=(0, (5, 3)),
        linewidth=2.0,
        alpha=0.7,
        zorder=1,
    )

    for base, method in _improvement_pairs(frame):
        axis.annotate(
            "",
            xy=(method["cost_per_1k"], method["mean_ood_pauroc_at_20"]),
            xytext=(base["cost_per_1k"], base["mean_ood_pauroc_at_20"]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": ORIGIN_PALETTE["Gleipnir"],
                "linestyle": (0, (3, 2)),
                "linewidth": 2.0,
                "mutation_scale": 14,
                "shrinkA": 9,
                "shrinkB": 9,
            },
            zorder=2,
        )
        # Inline label along the arrow, past its midpoint so it clears d1 in
        # the full figure. The tangent of a straight segment on log-x axes, in
        # data units at x, is (x * ln(x1 / x0), y1 - y0); the transform
        # rotates it to display.
        x0, x1 = base["cost_per_1k"], method["cost_per_1k"]
        y0, y1 = base["mean_ood_pauroc_at_20"], method["mean_ood_pauroc_at_20"]
        x_mid = x0 * (x1 / x0) ** DISTILLATION_LABEL_POSITION
        y_mid = y0 + DISTILLATION_LABEL_POSITION * (y1 - y0)
        angle = np.degrees(np.arctan2(y1 - y0, x_mid * np.log(x1 / x0)))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        axis.text(
            x_mid,
            y_mid,
            "soft distillation",
            rotation=angle,
            transform_rotates_text=True,
            rotation_mode="anchor",
            fontsize=10.5,
            fontweight="bold",
            color=ORIGIN_PALETTE["Gleipnir"],
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
            zorder=2.5,
        )

    cost_ratio = _draw_cost_callout(axis, frame)
    # Direction cue in the empty upper-left corner.
    axis.annotate(
        "Better",
        xy=(0.015, 0.975),
        xytext=(0.085, 0.86),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=13,
        fontweight="bold",
        color="#6B7280",
        ha="center",
        va="center",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#6B7280",
            "linewidth": 1.8,
            "mutation_scale": 16,
            "shrinkA": 12,
            "shrinkB": 0,
        },
        zorder=2,
    )

    committed = frame["pricing"].eq("committed")
    for category, group in frame.loc[~committed].groupby("category", sort=False):
        sns.scatterplot(
            data=group,
            x="cost_per_1k",
            y="mean_ood_pauroc_at_20",
            color=CATEGORY_PALETTE[category],
            marker=CATEGORY_MARKERS[category],
            s=92,
            alpha=0.78,
            edgecolor="#374151",
            linewidth=0.65,
            legend=False,
            ax=axis,
            zorder=2,
        )
        group_frontier = group.loc[group["computed_frontier"]]
        sns.scatterplot(
            data=group_frontier,
            x="cost_per_1k",
            y="mean_ood_pauroc_at_20",
            color=CATEGORY_PALETTE[category],
            marker=CATEGORY_MARKERS[category],
            s=165,
            edgecolor="#111827",
            linewidth=1.5,
            legend=False,
            ax=axis,
            zorder=3,
        )
    for _, row in frame.loc[committed].iterrows():
        axis.scatter(
            row["cost_per_1k"],
            row["mean_ood_pauroc_at_20"],
            marker="D",
            s=165 if row["computed_frontier"] else 92,
            facecolor="white",
            edgecolor=ORIGIN_PALETTE["Gleipnir"],
            linewidth=2.2,
            zorder=3,
        )

    labelled = frame.loc[
        frame["computed_frontier"]
        | frame["origin"].eq("Gleipnir")
        | frame["monitor"].isin(PAPER_COMPARISON_MONITORS)
    ].sort_values("cost_per_1k")
    if not full:
        labelled = labelled.loc[~labelled["monitor"].isin(MAIN_FIGURE_UNLABELLED)]
    label_bbox = {
        "boxstyle": "round,pad=0.12",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.84,
    }
    labels = []
    target_x = []
    target_y = []
    for label_index, (_, row) in enumerate(labelled.iterrows()):
        is_frontier = bool(row["computed_frontier"])
        monitor = str(row["monitor"])
        if monitor == "Qwen3.5-4B base":
            axis.annotate(
                _display_label(monitor),
                (row["cost_per_1k"], row["mean_ood_pauroc_at_20"]),
                xytext=(10, -10),
                textcoords="offset points",
                fontsize=11.5,
                fontweight="bold",
                color="#111827",
                ha="left",
                va="top",
                bbox=label_bbox,
                zorder=4,
            )
            continue
        if monitor in MANUAL_COMPARISON_LABELS:
            placement = MANUAL_COMPARISON_LABELS[monitor]
            axis.annotate(
                _display_label(monitor),
                (row["cost_per_1k"], row["mean_ood_pauroc_at_20"]),
                xytext=placement["xytext"],
                textcoords="offset points",
                fontsize=11.5,
                fontweight="bold",
                color="#111827" if is_frontier else "#4B5563",
                ha=placement["ha"],
                va=placement["va"],
                bbox=label_bbox,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#9CA3AF",
                    "linewidth": 0.7,
                },
                zorder=4,
            )
            continue
        vertical_offset = 0.018 if label_index % 2 == 0 else -0.018
        labels.append(
            axis.text(
                row["cost_per_1k"] * 1.02,
                row["mean_ood_pauroc_at_20"] + vertical_offset,
                _display_label(monitor),
                fontsize=11.5,
                fontweight="bold",
                color="#111827" if is_frontier else "#4B5563",
                ha="center",
                va="center",
                bbox=label_bbox,
                zorder=4,
            )
        )
        target_x.append(float(row["cost_per_1k"]))
        target_y.append(float(row["mean_ood_pauroc_at_20"]))
    obstacle_x, obstacle_y = _label_obstacles(frame)
    random_state = np.random.get_state()
    np.random.seed(20260830)
    try:
        adjust_text(
            labels,
            x=obstacle_x,
            y=obstacle_y,
            target_x=np.asarray(target_x),
            target_y=np.asarray(target_y),
            ax=axis,
            expand=(1.3, 1.5),
            force_text=(0.75, 1.1),
            force_static=(0.65, 0.95),
            force_pull=(0.01, 0.015),
            max_move=(32, 32),
            min_arrow_len=4,
            iter_lim=1000,
            prevent_crossings=True,
            arrowprops={
                "arrowstyle": "-",
                "color": "#9CA3AF",
                "linewidth": 0.7,
            },
        )
    finally:
        np.random.set_state(random_state)
    axis.set_xlabel("Inference cost (USD per 1,000 evaluations; log scale)")
    axis.set_ylabel("Detection performance\n(pAUROC@20, mean over OOD sources)")
    # The title states the finding; the subtitle says what is plotted.
    axis.set_title(
        "A distilled decision monitor beats the best reasoning monitor"
        f" at ~{_one_significant_figure(cost_ratio):g}× lower cost"
        if cost_ratio
        else "Monitoring cost and performance on tool trajectories",
        loc="left",
        fontsize=20.5,
        fontweight="bold",
        pad=23,
    )
    axis.text(
        0.0,
        1.015,
        "Monitoring agent tool trajectories for sabotage · mean over six held-out"
        " (OOD) AI control benchmarks, e.g. SHADE-Arena and AgentDojo",
        transform=axis.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color="#4B5563",
    )
    axis.grid(which="major", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    axis.grid(which="minor", axis="x", color="#E5E7EB", linewidth=0.5, alpha=0.5)

    # The Sinha et al. entry shows its points and frontier together.
    legend_handles = [
        Line2D(
            [],
            [],
            marker=CATEGORY_MARKERS[category],
            linestyle=(0, (1.5, 2.4)) if category == "Paper" else "none",
            color=color,
            linewidth=2.0,
            markerfacecolor=color,
            markeredgecolor="#374151",
            markeredgewidth=0.65,
            markersize=10.5,
            label=CATEGORY_LABELS[category],
        )
        for category in LEGEND_ORDER
        for color in [CATEGORY_PALETTE[category]]
    ]
    # The frontier entries form their own titled legend under the main one,
    # so the legend explains the dashed line and the asterisks.
    frontier_handles = [
        Line2D(
            [],
            [],
            color="#374151",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#111827",
            linewidth=2.5,
            markersize=10.5,
            label="Self-hosted at on-demand GPU rates",
        ),
        Line2D(
            [],
            [],
            color="#374151",
            linestyle=(0, (5, 3)),
            marker="D",
            markerfacecolor="white",
            markeredgecolor=ORIGIN_PALETTE["Gleipnir"],
            markeredgewidth=2.0,
            linewidth=2.2,
            markersize=10.5,
            label="Self-hosted at committed GPU rates*",
        ),
    ]
    legend_style = {
        "frameon": True,
        "prop": {"size": 13.5, "weight": "bold"},
        "handlelength": 2.2,
        "labelspacing": 0.65,
        "title_fontproperties": {"size": 13.5, "weight": "bold"},
        "alignment": "left",
    }
    figure.tight_layout()
    # Stack the category legend above the frontier legend, both right-aligned
    # and as wide as the wider of the two.
    renderer = figure.canvas.get_renderer()

    def axes_box(legend):
        return legend.get_window_extent(renderer).transformed(axis.transAxes.inverted())

    frontier_legend = axis.legend(
        handles=frontier_handles, loc="lower right", title="Frontier", **legend_style
    )
    category_legend = axis.legend(
        handles=legend_handles, loc="lower right", title="Monitor type", **legend_style
    )
    frontier_box, category_box = axes_box(frontier_legend), axes_box(category_legend)
    width = max(frontier_box.width, category_box.width)
    frontier_legend = axis.legend(
        handles=frontier_handles,
        title="Frontier",
        loc="lower left",
        bbox_to_anchor=(frontier_box.x1 - width, frontier_box.y0, width, 0),
        mode="expand",
        borderaxespad=0,
        **legend_style,
    )
    axis.add_artist(frontier_legend)
    axis.legend(
        handles=legend_handles,
        title="Monitor type",
        loc="lower left",
        bbox_to_anchor=(frontier_box.x1 - width, frontier_box.y1 + 0.012, width, 0),
        mode="expand",
        borderaxespad=0,
        **legend_style,
    )
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Markdown registry containing the canonical reference table",
    )
    parser.add_argument(
        "--output",
        type=Path,
        action="append",
        help=(
            "Output image path; repeat for multiple formats. Defaults to both "
            "PNG and SVG."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Plot every registry point (the appendix figure)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_frontier_registry(args.source)
    plotted = select_plot_points(frame, full=args.full)
    omitted_count = len(frame) - len(plotted)
    default_outputs = (
        FULL_OUTPUTS if args.full else [DEFAULT_OUTPUT, DEFAULT_VECTOR_OUTPUT]
    )
    outputs = args.output or default_outputs
    frontier_count = int(plotted["computed_frontier"].sum())
    for output_path in outputs:
        output = save_figure(plot_frontier(frame, full=args.full), output_path)
        print(
            f"Wrote {output} from {len(plotted)} plotted points "
            f"({frontier_count} frontier; {omitted_count} omitted)."
        )


if __name__ == "__main__":
    main()
