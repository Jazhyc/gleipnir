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
EXCLUDED_MONITORS = {
    "Claude Opus 4.6 prompted",
    "Qwen3.5-9B base (reasoned binary)",
}
GLEIPNIR_METHODS = {
    "Qwen3.5-4B Kimi-soft mixed": "Gleipnir 4B",
    "Qwen3.5-9B Kimi-soft mixed": "Gleipnir 9B",
}
BASE_TO_METHOD = {
    "Qwen3.5-4B base": "Qwen3.5-4B Kimi-soft mixed",
    "Qwen3.5-9B base": "Qwen3.5-9B Kimi-soft mixed",
}
PAPER_COMPARISON_MONITORS = {
    "Kimi-K2.6 SFT",
    "Qwen3.5-4B SFT+RL",
    "Qwen3.5-27B SFT+RL",
}
MANUAL_COMPARISON_LABELS = {
    "Kimi K3 binary logits": {
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
    "Qwen3.5-27B base": {"xytext": (0, 18), "ha": "center", "va": "bottom"},
    "Qwen3.5-27B SFT+RL": {"xytext": (0, -36), "ha": "center", "va": "top"},
    "Claude Sonnet 4.6 prompted": {
        "xytext": (0, -18),
        "ha": "center",
        "va": "top",
    },
    "Gemini 3.1 Pro prompted": {
        "xytext": (0, 18),
        "ha": "center",
        "va": "bottom",
    },
}
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
    frame["computed_frontier"] = pareto_frontier_mask(
        frame["cost_per_1k"].to_numpy(),
        frame["mean_ood_pauroc_at_20"].to_numpy(),
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


def _display_label(monitor: str) -> str:
    replacements = {
        "Kimi K2.6 binary logits": "Kimi K2.6",
        "Kimi K3 binary logits": "Kimi K3",
        "Claude Sonnet 4.6 prompted": "Claude Sonnet 4.6",
        "Gemini 3.1 Pro prompted": "Gemini 3.1 Pro",
        "Claude Opus 4.6 prompted": "Claude Opus 4.6",
        "Qwen3.5-27B base": "Qwen3.5-27B",
        **GLEIPNIR_METHODS,
    }
    display = replacements.get(monitor, monitor)
    display = re.sub(r"\s+base(?=$|\s+\()", "", display)
    return textwrap.fill(display, width=22)


def select_plot_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply documented presentation-only omissions to registry points."""
    is_ours = frame["origin"].eq("Gleipnir")
    is_logit_interface = frame["interface"].str.contains("logits", case=False, na=False)
    included = (
        ~frame["monitor"].isin(EXCLUDED_MONITORS)
        & frame["mean_ood_pauroc_at_20"].ge(MIN_PLOTTED_PERFORMANCE)
        & (~is_ours | is_logit_interface)
    )
    return frame.loc[included].copy()


def _label_obstacles(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Sample points and frontier segments for label collision avoidance."""
    x_parts = [frame["cost_per_1k"].to_numpy()]
    y_parts = [frame["mean_ood_pauroc_at_20"].to_numpy()]
    frontier = frame.loc[frame["computed_frontier"]].sort_values("cost_per_1k")
    for (_, start), (_, end) in zip(
        frontier.iloc[:-1].iterrows(),
        frontier.iloc[1:].iterrows(),
        strict=True,
    ):
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


def plot_frontier(frame: pd.DataFrame) -> plt.Figure:
    """Build the publication-ready OOD frontier plot."""
    frame = select_plot_points(frame)
    set_plot_style()
    figure, axis = plt.subplots(figsize=(14.2, 8.5))
    axis.set_xscale("log")
    axis.set_xticks([0.3, 1.0, 3.0, 10.0, 30.0, 80.0])
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_xlim(
        frame["cost_per_1k"].min() / 1.35,
        max(80.0, frame["cost_per_1k"].max() * 1.06),
    )
    axis.set_ylim(MIN_PLOTTED_PERFORMANCE - 0.02, 0.99)

    frontier = frame.loc[frame["computed_frontier"]].sort_values("cost_per_1k")
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

    markers = {"Paper": "o", "Gleipnir": "D"}
    for origin, group in frame.groupby("origin", sort=False):
        sns.scatterplot(
            data=group,
            x="cost_per_1k",
            y="mean_ood_pauroc_at_20",
            color=ORIGIN_PALETTE.get(origin, "#777777"),
            marker=markers.get(origin, "o"),
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
            color=ORIGIN_PALETTE.get(origin, "#777777"),
            marker=markers.get(origin, "o"),
            s=165,
            edgecolor="#111827",
            linewidth=1.5,
            legend=False,
            ax=axis,
            zorder=3,
        )

    labelled = frame.loc[
        frame["computed_frontier"]
        | frame["origin"].eq("Gleipnir")
        | frame["monitor"].isin(PAPER_COMPARISON_MONITORS)
    ].sort_values("cost_per_1k")
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
    axis.set_ylabel("Mean-OOD pAUROC@20")
    axis.set_title(
        "Logit-based monitors on the tool-trajectory cost–performance frontier",
        loc="left",
        fontsize=20.5,
        fontweight="bold",
        pad=28,
    )
    axis.text(
        0.0,
        1.015,
        "Strict six-source OOD suite · lower cost and higher performance are better",
        transform=axis.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color="#4B5563",
    )
    axis.grid(which="major", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    axis.grid(which="minor", axis="x", color="#E5E7EB", linewidth=0.5, alpha=0.5)

    origin_labels = {"Paper": "Sinha et al.", "Gleipnir": "Ours (logits)"}
    legend_handles = [
        Line2D(
            [],
            [],
            marker=markers[origin],
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="#374151",
            markeredgewidth=0.65,
            markersize=10.5,
            label=origin_labels.get(origin, origin),
        )
        for origin, color in ORIGIN_PALETTE.items()
    ]
    legend_handles.append(
        Line2D(
            [],
            [],
            color="#374151",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#111827",
            linewidth=2.5,
            markersize=10.5,
            label="Current frontier",
        )
    )
    legend_handles.append(
        Line2D(
            [],
            [],
            color=ORIGIN_PALETTE["Paper"],
            linestyle=(0, (1.5, 2.4)),
            linewidth=2.0,
            label="Paper-only frontier",
        )
    )
    axis.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        prop={"size": 13.5, "weight": "bold"},
        handlelength=2.2,
        labelspacing=0.65,
    )
    figure.tight_layout()
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_frontier_registry(args.source)
    plotted = select_plot_points(frame)
    omitted_count = len(frame) - len(plotted)
    outputs = args.output or [DEFAULT_OUTPUT, DEFAULT_VECTOR_OUTPUT]
    frontier_count = int(plotted["computed_frontier"].sum())
    for output_path in outputs:
        output = save_figure(plot_frontier(frame), output_path)
        print(
            f"Wrote {output} from {len(plotted)} plotted points "
            f"({frontier_count} frontier; {omitted_count} omitted)."
        )


if __name__ == "__main__":
    main()
