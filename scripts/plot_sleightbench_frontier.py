"""Plot the SleightBench cost frontier for the technical report.

Reads ``results/sleightbench_eval/frontier.json`` (written by
``python -m experiments.sleightbench_eval.frontier``) and draws it in the style
of the report's OOD frontier, with the 3% cost tolerance. Only the
committed-use Gleipnir rows are plotted, with one frontier; the base models
the superseded pplx-decider v1.2 and Jev 1.13, and GPT-5.5 are left out
(user, 2026-11-19). The appendix table keeps them.
"""

from __future__ import annotations

import argparse
import json
import os
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

from gleipnir.plotting import pareto_frontier_mask, save_figure, set_plot_style
from scripts.plot_tool_trajectory_ood_frontier import (
    CATEGORY_LABELS,
    CATEGORY_MARKERS,
    CATEGORY_PALETTE,
    FRONTIER_COST_TOLERANCE,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY_ROOT / "results/sleightbench_eval/frontier.json"
ASSETS = REPOSITORY_ROOT / "outputs/reports/gleipnir_v1_technical_report/assets"
METRICS = {
    "pauroc_at_20": ("SleightBench pAUROC@20", "sleightbench_frontier", (0.15, 0.80)),
    "catch_rate": (
        "SleightBench catch rate at 1% FPR",
        "sleightbench_catch_frontier",
        (-0.03, 0.62),
    ),
}
# Superseded by newer versions of the same models, and GPT-5.5, an older model
# scored as a check against the authors (user, 2026-11-19); the table keeps
# them.
OMITTED = {"pplx-decider v1.2", "Jev 1.13", "GPT-5.5"}
# Label placement per metric: offset in points, horizontal and vertical
# alignment. Labels default to the right of their point.
LABELS = {
    "pauroc_at_20": {
        "Gleipnir 4B": ((0, -13), "center", "top"),
        "Gleipnir 27B": ((-10, 0), "right", "center"),
        # Shares its point with Gleipnir 27B.
        "pplx-decider v1.2": ((8, 6), "left", "bottom"),
        "pplx-decider v1.3": ((0, 12), "center", "bottom"),
        "Jev 1.14": ((0, -11), "center", "top"),
        # Left of its point, clear of Gemini Decision 1 to its right.
        "Drex Max": ((-10, 0), "right", "center"),
        "GPT-6 Luna": ((0, -11), "center", "top"),
        "Gemini Decision 1": ((0, 10), "center", "bottom"),
        "GPT-6.1 Terra": ((0, 12), "center", "bottom"),
        "GPT-6 Sol": ((0, 12), "center", "bottom"),
        "Kimi K3": ((6, -12), "right", "top"),
        "GPT-5.5": ((0, 12), "center", "bottom"),
    },
    "catch_rate": {
        "Gleipnir 4B": ((0, -13), "center", "top"),
        "Gleipnir 27B": ((-10, 0), "right", "center"),
        "pplx-decider v1.2": ((0, -11), "center", "top"),
        "pplx-decider v1.3": ((0, 12), "center", "bottom"),
        "GPT-6.1 Terra": ((0, 12), "center", "bottom"),
        "GPT-6 Sol": ((0, 12), "center", "bottom"),
        "Kimi K3": ((6, 10), "right", "bottom"),
        "GPT-5.5": ((6, -11), "right", "top"),
    },
}


def load(path: Path = SOURCE) -> pd.DataFrame:
    """The plotted rows: list-priced monitors and committed-use Gleipnir."""
    frame = pd.DataFrame(json.loads(path.read_text())["monitors"])
    frame = frame.loc[
        frame["pricing"].ne("on_demand")
        & frame["kind"].ne("base")
        & ~frame["monitor"].isin(OMITTED)
    ].reset_index(drop=True)
    frame["category"] = np.where(frame["kind"].eq("gleipnir"), "Gleipnir", "Other")
    return frame


def with_frontier(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["frontier"] = pareto_frontier_mask(
        frame["cost_per_1k"].to_numpy(),
        frame[metric].to_numpy(),
        cost_tolerance=FRONTIER_COST_TOLERANCE,
    )
    return frame


def _label(monitor: str) -> str:
    name = monitor.replace("pplx-decider", "Decider")
    return "Kimi K3 (teacher)" if name == "Kimi K3" else name


def plot(frame: pd.DataFrame, metric: str) -> plt.Figure:
    ylabel, _, ylim = METRICS[metric]
    frame = with_frontier(frame, metric)
    set_plot_style()
    figure, axis = plt.subplots(figsize=(10.5, 6.6))
    axis.set_xscale("log")
    axis.set_xticks([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_xlim(frame["cost_per_1k"].min() / 1.35, frame["cost_per_1k"].max() * 1.3)
    axis.set_ylim(*ylim)

    line = frame.loc[frame["frontier"]].sort_values("cost_per_1k")
    axis.plot(
        line["cost_per_1k"],
        line[metric],
        color="#374151",
        linewidth=2.3,
        alpha=0.82,
        zorder=1,
    )
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
    for _, row in frame.iterrows():
        on_frontier = bool(row["frontier"])
        axis.scatter(
            row["cost_per_1k"],
            row[metric],
            marker=CATEGORY_MARKERS[row["category"]],
            s=165 if on_frontier else 92,
            facecolor=CATEGORY_PALETTE[row["category"]],
            alpha=1.0 if on_frontier else 0.78,
            edgecolor="#111827" if on_frontier else "#374151",
            linewidth=1.5 if on_frontier else 0.65,
            zorder=3 if on_frontier else 2,
        )
        offset, ha, va = LABELS[metric].get(row["monitor"], ((10, 0), "left", "center"))
        axis.annotate(
            _label(row["monitor"]),
            (row["cost_per_1k"], row[metric]),
            xytext=offset,
            textcoords="offset points",
            fontsize=11.5,
            fontweight="bold",
            color="#111827" if on_frontier else "#4B5563",
            ha=ha,
            va=va,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.84,
            },
            zorder=4,
        )

    axis.set_xlabel("Inference cost (USD per 1,000 evaluations)", fontsize=14)
    axis.set_ylabel(ylabel, fontsize=14)
    axis.grid(which="major", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    axis.grid(which="minor", axis="x", color="#E5E7EB", linewidth=0.5, alpha=0.5)
    handles = [
        Line2D(
            [],
            [],
            marker=CATEGORY_MARKERS[category],
            linestyle="none",
            markerfacecolor=CATEGORY_PALETTE[category],
            markeredgecolor="#374151",
            markeredgewidth=0.65,
            markersize=10.5,
            label=CATEGORY_LABELS[category],
        )
        for category in ("Other", "Gleipnir")
    ] + [Line2D([], [], color="#374151", linewidth=2.5, label="Frontier")]
    axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        prop={"size": 13, "weight": "bold"},
        handlelength=2.2,
        columnspacing=1.6,
    )
    figure.tight_layout()
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=sorted(METRICS), default="pauroc_at_20")
    parser.add_argument("--output", type=Path, action="append")
    args = parser.parse_args()
    stem = METRICS[args.metric][1]
    outputs = args.output or [ASSETS / f"{stem}.pdf", ASSETS / f"{stem}.svg"]
    frame = load()
    for output in outputs:
        print(f"wrote {save_figure(plot(frame, args.metric), output)}")


if __name__ == "__main__":
    main()
