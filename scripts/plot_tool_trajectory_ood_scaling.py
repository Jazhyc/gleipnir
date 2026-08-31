"""Plot the canonical Qwen3.5-9B soft-distillation OOD scaling curve."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gleipnir-matplotlib")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter

from gleipnir.plotting import ORIGIN_PALETTE, save_figure, set_plot_style

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    REPOSITORY_ROOT / "docs/research/tool_trajectory_distillation_ood_scaling.md"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "figures/tool_trajectory_ood_data_scaling.png"
DEFAULT_VECTOR_OUTPUT = (
    REPOSITORY_ROOT / "figures/tool_trajectory_ood_data_scaling.svg"
)
EXPECTED_COLUMNS = [
    "Backbone",
    "Training data",
    "Rows",
    "Mean-OOD pAUROC@20",
    "Mean-OOD AUROC",
    "Pooled pAUROC@20",
    "Pooled AUROC",
]
MATCHED_ROW_SCHEDULE = [204, 504, 996, 2_004, 4_008, 8_688]
METRICS = {
    "mean_ood_pauroc_at_20": {
        "label": "Mean-OOD pAUROC@20",
        "color": ORIGIN_PALETTE["Gleipnir"],
        "marker": "D",
        "linestyle": "-",
    },
    "mean_ood_auroc": {
        "label": "Mean-OOD AUROC",
        "color": ORIGIN_PALETTE["Paper"],
        "marker": "o",
        "linestyle": (0, (5, 2)),
    },
}


def _markdown_cells(line: str) -> list[str]:
    cells = re.split(r"(?<!\\)\|", line.strip())
    if cells and not cells[0]:
        cells = cells[1:]
    if cells and not cells[-1]:
        cells = cells[:-1]
    return [cell.strip().replace(r"\|", "|") for cell in cells]


def _plain_text(value: str) -> str:
    return value.replace("**", "").replace("`", "").strip()


def load_scaling_results(path: str | Path) -> pd.DataFrame:
    """Load and validate the matched Qwen3.5-9B rows in the results table."""
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
        raise ValueError(f"results table header not found in {path}")

    rows: list[list[str]] = []
    for line in lines[header_index + 2 :]:
        if not line.strip().startswith("|"):
            break
        cells = _markdown_cells(line)
        if len(cells) != len(EXPECTED_COLUMNS):
            raise ValueError(f"malformed results-table row: {line}")
        rows.append([_plain_text(cell) for cell in cells])
    if not rows:
        raise ValueError(f"results table has no data rows in {path}")

    frame = pd.DataFrame(rows, columns=EXPECTED_COLUMNS).rename(
        columns={
            "Backbone": "backbone",
            "Training data": "training_data",
            "Rows": "rows",
            "Mean-OOD pAUROC@20": "mean_ood_pauroc_at_20",
            "Mean-OOD AUROC": "mean_ood_auroc",
            "Pooled pAUROC@20": "pooled_pauroc_at_20",
            "Pooled AUROC": "pooled_auroc",
        }
    )
    frame["rows"] = pd.to_numeric(frame["rows"].str.replace(",", ""))
    metric_columns = [
        "mean_ood_pauroc_at_20",
        "mean_ood_auroc",
        "pooled_pauroc_at_20",
        "pooled_auroc",
    ]
    frame[metric_columns] = frame[metric_columns].apply(pd.to_numeric)
    matched = frame.loc[
        frame["backbone"].eq("Qwen3.5-9B")
        & frame["training_data"].eq("matched")
    ].sort_values("rows")

    if matched["rows"].tolist() != MATCHED_ROW_SCHEDULE:
        raise ValueError(
            "matched Qwen3.5-9B row schedule changed: "
            f"expected {MATCHED_ROW_SCHEDULE}, got {matched['rows'].tolist()}"
        )
    values = matched[metric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("scaling metrics must be finite values between zero and one")
    return matched.reset_index(drop=True)


def _annotate_peak(
    axis: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    offset: tuple[float, float],
) -> None:
    style = METRICS[metric]
    peak = frame.loc[frame[metric].idxmax()]
    x = float(peak["rows"])
    y = float(peak[metric])
    axis.scatter(
        [x],
        [y],
        s=205,
        marker=style["marker"],
        color=style["color"],
        edgecolor="#111827",
        linewidth=1.6,
        zorder=4,
    )
    short_label = "Best pAUROC@20" if metric.endswith("pauroc_at_20") else "Best AUROC"
    axis.annotate(
        f"{short_label}: {y:.1%}\n{x:,.0f} rows",
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=11.5,
        fontweight="bold",
        color="#111827",
        ha="left" if offset[0] >= 0 else "right",
        va="bottom" if offset[1] >= 0 else "top",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": style["color"],
            "linewidth": 1.0,
            "alpha": 0.94,
        },
        arrowprops={
            "arrowstyle": "-",
            "color": style["color"],
            "linewidth": 1.1,
        },
        zorder=5,
    )


def plot_scaling(frame: pd.DataFrame) -> plt.Figure:
    """Build the publication-ready matched-data OOD scaling plot."""
    set_plot_style()
    figure, axis = plt.subplots(figsize=(14.2, 8.5))
    axis.set_xscale("log")

    for metric, style in METRICS.items():
        axis.plot(
            frame["rows"],
            frame[metric],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.8,
            markersize=9.5,
            markeredgecolor="#374151",
            markeredgewidth=0.7,
            alpha=0.9,
            zorder=3,
        )

    _annotate_peak(axis, frame, "mean_ood_auroc", (18, 14))
    _annotate_peak(axis, frame, "mean_ood_pauroc_at_20", (-18, -18))

    axis.set_xticks(MATCHED_ROW_SCHEDULE)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_xlim(MATCHED_ROW_SCHEDULE[0] / 1.25, MATCHED_ROW_SCHEDULE[-1] * 1.2)
    axis.set_ylim(0.65, 0.98)
    axis.set_xlabel("Matched tool-trajectory training examples (log scale)")
    axis.set_ylabel("Source-balanced OOD performance")
    axis.set_title(
        "Qwen3.5-9B OOD performance scales non-monotonically with data",
        loc="left",
        fontsize=20.5,
        fontweight="bold",
        pad=18,
    )
    axis.text(
        0.0,
        1.015,
        (
            "Strict six-source OOD suite · one epoch · one seed per point · "
            "higher is better"
        ),
        transform=axis.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color="#4B5563",
    )
    axis.grid(which="major", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    axis.grid(which="minor", axis="x", color="#E5E7EB", linewidth=0.5, alpha=0.5)

    legend_handles = [
        Line2D(
            [],
            [],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markeredgecolor="#374151",
            markeredgewidth=0.7,
            linewidth=2.8,
            markersize=9.5,
            label=style["label"],
        )
        for style in METRICS.values()
    ]
    axis.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        prop={"size": 13.5, "weight": "bold"},
        handlelength=2.6,
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
        help="Markdown research note containing the canonical results table",
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
    frame = load_scaling_results(args.source)
    outputs = args.output or [DEFAULT_OUTPUT, DEFAULT_VECTOR_OUTPUT]
    for output in outputs:
        save_figure(plot_scaling(frame), output)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
