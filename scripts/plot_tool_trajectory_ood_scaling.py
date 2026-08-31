"""Compare Gleipnir's 9B OOD scaling with published 4B and 27B curves."""

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
PAPER_EXPECTED_COLUMNS = [
    "Backbone",
    "Rows",
    "Mean-OOD pAUROC@20 (digitized)",
]
MATCHED_ROW_SCHEDULE = [204, 504, 996, 2_004, 4_008, 8_688]
SERIES_STYLES = {
    "ours_9b": {
        "label": "Ours · 9B soft logits",
        "color": ORIGIN_PALETTE["Gleipnir"],
        "marker": "D",
        "linestyle": "-",
        "linewidth": 3.0,
    },
    "paper_4b": {
        "label": "Sinha et al. · 4B rationale SFT",
        "color": "#7CA4D1",
        "marker": "o",
        "linestyle": (0, (5, 2)),
        "linewidth": 2.4,
    },
    "paper_27b": {
        "label": "Sinha et al. · 27B rationale SFT",
        "color": ORIGIN_PALETTE["Paper"],
        "marker": "s",
        "linestyle": (0, (5, 2)),
        "linewidth": 2.6,
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


def _table_rows(path: str | Path, expected_columns: list[str]) -> list[list[str]]:
    """Return plain-text rows from the uniquely identified Markdown table."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if _markdown_cells(line) == expected_columns
        ),
        None,
    )
    if header_index is None:
        raise ValueError(f"table header not found in {path}: {expected_columns}")

    rows: list[list[str]] = []
    for line in lines[header_index + 2 :]:
        if not line.strip().startswith("|"):
            break
        cells = _markdown_cells(line)
        if len(cells) != len(expected_columns):
            raise ValueError(f"malformed results-table row: {line}")
        rows.append([_plain_text(cell) for cell in cells])
    if not rows:
        raise ValueError(f"results table has no data rows in {path}")
    return rows


def load_scaling_results(path: str | Path) -> pd.DataFrame:
    """Load and validate the matched Qwen3.5-9B rows in the results table."""
    rows = _table_rows(path, EXPECTED_COLUMNS)

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


def load_paper_scaling_reference(path: str | Path) -> pd.DataFrame:
    """Load the vector-digitized Sinha et al. Figure 7 reference curves."""
    rows = _table_rows(path, PAPER_EXPECTED_COLUMNS)
    frame = pd.DataFrame(rows, columns=PAPER_EXPECTED_COLUMNS).rename(
        columns={
            "Backbone": "backbone",
            "Rows": "rows",
            "Mean-OOD pAUROC@20 (digitized)": "mean_ood_pauroc_at_20",
        }
    )
    frame["rows"] = pd.to_numeric(frame["rows"].str.replace(",", ""))
    frame["mean_ood_pauroc_at_20"] = pd.to_numeric(
        frame["mean_ood_pauroc_at_20"]
    )
    if set(frame["backbone"]) != {"Qwen3.5-4B", "Qwen3.5-27B"}:
        raise ValueError("paper reference must contain Qwen3.5-4B and Qwen3.5-27B")
    for backbone, group in frame.groupby("backbone"):
        schedule = group.sort_values("rows")["rows"].tolist()
        if schedule != MATCHED_ROW_SCHEDULE:
            raise ValueError(
                f"paper {backbone} row schedule changed: "
                f"expected {MATCHED_ROW_SCHEDULE}, got {schedule}"
            )
    values = frame["mean_ood_pauroc_at_20"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("paper scaling metrics must be finite values from zero to one")
    return frame.sort_values(["backbone", "rows"]).reset_index(drop=True)


def _annotate_our_peak(axis: plt.Axes, frame: pd.DataFrame) -> None:
    style = SERIES_STYLES["ours_9b"]
    metric = "mean_ood_pauroc_at_20"
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
    axis.annotate(
        f"Our best: {y:.1%}\n{x:,.0f} rows",
        xy=(x, y),
        xytext=(-18, 18),
        textcoords="offset points",
        fontsize=11.5,
        fontweight="bold",
        color="#111827",
        ha="right",
        va="bottom",
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


def _plot_series(
    axis: plt.Axes,
    frame: pd.DataFrame,
    style_key: str,
) -> None:
    style = SERIES_STYLES[style_key]
    axis.plot(
        frame["rows"],
        frame["mean_ood_pauroc_at_20"],
        label=style["label"],
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        markersize=9.5,
        markerfacecolor=style["color"] if style_key == "ours_9b" else "white",
        markeredgecolor="#374151",
        markeredgewidth=0.8,
        alpha=0.94,
        zorder=3 if style_key == "ours_9b" else 2,
    )


def _label_endpoint(
    axis: plt.Axes,
    frame: pd.DataFrame,
    style_key: str,
    vertical_offset: float,
) -> None:
    style = SERIES_STYLES[style_key]
    endpoint = frame.sort_values("rows").iloc[-1]
    axis.annotate(
        style["label"],
        xy=(float(endpoint["rows"]), float(endpoint["mean_ood_pauroc_at_20"])),
        xytext=(28, vertical_offset),
        textcoords="offset points",
        fontsize=11.5,
        fontweight="bold",
        color=style["color"],
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
        arrowprops={
            "arrowstyle": "-",
            "color": style["color"],
            "linewidth": 0.9,
        },
        zorder=5,
    )


def plot_scaling(frame: pd.DataFrame, paper: pd.DataFrame) -> plt.Figure:
    """Build the publication-ready OOD pAUROC data-scaling comparison."""
    set_plot_style()
    figure, axis = plt.subplots(figsize=(14.2, 8.5))
    axis.set_xscale("log")

    paper_4b = paper.loc[paper["backbone"].eq("Qwen3.5-4B")]
    paper_27b = paper.loc[paper["backbone"].eq("Qwen3.5-27B")]
    _plot_series(axis, paper_4b, "paper_4b")
    _plot_series(axis, paper_27b, "paper_27b")
    _plot_series(axis, frame, "ours_9b")

    _annotate_our_peak(axis, frame)
    _label_endpoint(axis, paper_4b, "paper_4b", 0)
    _label_endpoint(axis, paper_27b, "paper_27b", -13)
    _label_endpoint(axis, frame, "ours_9b", 13)

    axis.set_xticks(MATCHED_ROW_SCHEDULE)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_xlim(MATCHED_ROW_SCHEDULE[0] / 1.25, 18_000)
    axis.set_ylim(0.60, 0.90)
    axis.set_xlabel("Tool-trajectory training examples (log scale)")
    axis.set_ylabel("Mean-OOD pAUROC@20")
    axis.set_title(
        "Soft-logit 9B crosses the paper's 27B rationale-SFT curve near 1K rows",
        loc="left",
        fontsize=20.5,
        fontweight="bold",
        pad=18,
    )
    axis.text(
        0.0,
        1.015,
        (
            "Same row counts, OOD suite, and metric · different supervision, "
            "training budgets, and checkpoint selection"
        ),
        transform=axis.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color="#4B5563",
    )
    axis.grid(which="major", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    axis.grid(which="minor", axis="x", color="#E5E7EB", linewidth=0.5, alpha=0.5)
    axis.text(
        0.0,
        -0.16,
        (
            "Paper reference values are vector-digitized from Sinha et al. "
            "Figure 7 and rounded to 0.001."
        ),
        transform=axis.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color="#6B7280",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
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
    paper = load_paper_scaling_reference(args.source)
    outputs = args.output or [DEFAULT_OUTPUT, DEFAULT_VECTOR_OUTPUT]
    for output in outputs:
        save_figure(plot_scaling(frame, paper), output)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
