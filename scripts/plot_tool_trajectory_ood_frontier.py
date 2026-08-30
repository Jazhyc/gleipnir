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
EXCLUDED_MONITORS = {"Claude Opus 4.6 prompted"}
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
    frame["mean_ood_pauroc_at_20"] = pd.to_numeric(
        frame["mean_ood_pauroc_at_20"]
    )
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
        "Kimi K3 binary logits": "Kimi K3 logits",
        "Claude Sonnet 4.6 prompted": "Claude Sonnet 4.6",
        "Gemini 3.1 Pro prompted": "Gemini 3.1 Pro",
        "Claude Opus 4.6 prompted": "Claude Opus 4.6",
    }
    return textwrap.fill(replacements.get(monitor, monitor), width=22)


def select_plot_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply documented presentation-only omissions to registry points."""
    return frame.loc[~frame["monitor"].isin(EXCLUDED_MONITORS)].copy()


def plot_frontier(frame: pd.DataFrame) -> plt.Figure:
    """Build the publication-ready OOD frontier plot."""
    frame = select_plot_points(frame)
    set_plot_style()
    figure, axis = plt.subplots(figsize=(13.2, 8.0))
    axis.set_xscale("log")
    axis.set_xticks([0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_xlim(
        frame["cost_per_1k"].min() / 1.35,
        frame["cost_per_1k"].max() * 1.45,
    )
    axis.set_ylim(0.44, 0.985)

    frontier = frame.loc[frame["computed_frontier"]].sort_values("cost_per_1k")
    axis.plot(
        frontier["cost_per_1k"],
        frontier["mean_ood_pauroc_at_20"],
        color="#374151",
        linewidth=2.3,
        alpha=0.82,
        zorder=1,
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
        frame["computed_frontier"] | frame["origin"].eq("Gleipnir")
    ].sort_values("cost_per_1k")
    labels = []
    for label_index, (_, row) in enumerate(labelled.iterrows()):
        is_frontier = bool(row["computed_frontier"])
        vertical_offset = 0.014 if label_index % 2 == 0 else -0.014
        labels.append(
            axis.text(
                row["cost_per_1k"] * 1.02,
                row["mean_ood_pauroc_at_20"] + vertical_offset,
                _display_label(str(row["monitor"])),
                fontsize=9.25,
                fontweight="bold" if row["origin"] == "Gleipnir" else "normal",
                color="#111827" if is_frontier else "#4B5563",
                ha="center",
                va="center",
                zorder=4,
            )
        )
    adjust_text(
        labels,
        x=frame["cost_per_1k"].to_numpy(),
        y=frame["mean_ood_pauroc_at_20"].to_numpy(),
        target_x=labelled["cost_per_1k"].to_numpy(),
        target_y=labelled["mean_ood_pauroc_at_20"].to_numpy(),
        ax=axis,
        expand=(1.2, 1.35),
        force_text=(0.55, 0.9),
        force_static=(0.45, 0.7),
        force_pull=(0.01, 0.015),
        max_move=(24, 24),
        min_arrow_len=4,
        iter_lim=600,
        prevent_crossings=True,
        arrowprops={"arrowstyle": "-", "color": "#9CA3AF", "linewidth": 0.7},
    )
    axis.set_xlabel("Inference cost (USD per 1,000 evaluations; log scale)")
    axis.set_ylabel("Mean-OOD pAUROC@20")
    axis.set_title(
        "Tool-trajectory monitor cost–performance frontier",
        loc="left",
        fontsize=19,
        fontweight="bold",
        pad=18,
    )
    axis.text(
        0.0,
        1.015,
        "Strict six-source OOD suite · lower cost and higher performance are better",
        transform=axis.transAxes,
        fontsize=10.5,
        color="#4B5563",
    )
    axis.grid(which="major", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    axis.grid(which="minor", axis="x", color="#E5E7EB", linewidth=0.5, alpha=0.5)

    origin_labels = {"Paper": "Sinha et al.", "Gleipnir": "Ours"}
    legend_handles = [
        Line2D(
            [],
            [],
            marker=markers[origin],
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="#374151",
            markeredgewidth=0.65,
            markersize=8,
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
            linewidth=2,
            markersize=8,
            label="Pareto frontier",
        )
    )
    axis.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=10)
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
