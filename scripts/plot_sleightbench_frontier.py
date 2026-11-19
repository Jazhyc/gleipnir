"""Plot the SleightBench cost frontier for the technical report.

Reads ``results/sleightbench_eval/frontier.json`` (written by
``python -m experiments.sleightbench_eval.frontier``) and draws it in the style
of the report's OOD frontier: one frontier per Gleipnir pricing regime, the
3% cost tolerance, and arrows from each base model to its Gleipnir monitor.
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

from gleipnir.plotting import (
    ORIGIN_PALETTE,
    pareto_frontier_mask,
    save_figure,
    set_plot_style,
)
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
BASE_TO_GLEIPNIR = {"Qwen3.5-4B": "Gleipnir 4B", "Qwen3.8-27B": "Gleipnir 27B"}
# Label placement (offset in points, horizontal and vertical alignment), per
# metric; labels default to the right of their point.
LABELS = {
    "pauroc_at_20": {
        ("Gleipnir 4B", "committed"): ((0, 12), "center", "bottom"),
        ("Gleipnir 4B", "on_demand"): ((0, -13), "center", "top"),
        ("Gleipnir 27B", "committed"): ((-10, 4), "right", "bottom"),
        ("Gleipnir 27B", "on_demand"): ((1.08, 0.343), "left", "bottom", "data"),
        ("pplx-decider v1.3", "list"): ((0, 12), "center", "bottom"),
        # Shares its point with the committed 27B.
        ("pplx-decider v1.2", "list"): ((0.40, 0.335), "left", "bottom", "data"),
        ("Qwen3.5-4B", "list"): ((10, -2), "left", "top"),
        ("Qwen3.8-27B", "list"): ((10, 0), "left", "center"),
        ("Jev 1.13", "list"): ((0.86, 0.226), "left", "center", "data"),
        ("GPT-6 Luna", "list"): ((1.7, 0.222), "left", "center", "data"),
        ("Gemini Decision 1", "list"): ((10, 0), "left", "center"),
        ("GPT-6.1 Terra", "list"): ((0, 12), "center", "bottom"),
        ("GPT-6 Sol", "list"): ((0, 12), "center", "bottom"),
        ("Kimi K3", "list"): ((0, -12), "center", "top"),
        ("GPT-5.5", "list"): ((0, 12), "center", "bottom"),
    },
    "catch_rate": {
        ("Gleipnir 4B", "committed"): ((0, 12), "center", "bottom"),
        ("Gleipnir 4B", "on_demand"): ((0, -13), "center", "top"),
        ("Gleipnir 27B", "committed"): ((-10, 0), "right", "center"),
        ("Gleipnir 27B", "on_demand"): ((0, 10), "center", "bottom"),
        ("pplx-decider v1.3", "list"): ((0, 12), "center", "bottom"),
        ("pplx-decider v1.2", "list"): ((0.40, 0.095), "left", "center", "data"),
        ("Qwen3.5-4B", "list"): ((10, -2), "left", "top"),
        ("Qwen3.8-27B", "list"): ((10, 0), "left", "center"),
        ("Jev 1.13", "list"): ((10, 0), "left", "center"),
        ("GPT-6 Luna", "list"): ((10, 0), "left", "center"),
        ("Gemini Decision 1", "list"): ((10, 0), "left", "center"),
        ("GPT-6.1 Terra", "list"): ((0, 12), "center", "bottom"),
        ("GPT-6 Sol", "list"): ((0, 12), "center", "bottom"),
        ("Kimi K3", "list"): ((0, 10), "center", "bottom"),
        ("GPT-5.5", "list"): ((6, -11), "right", "top"),
    },
}


def load(path: Path = SOURCE) -> pd.DataFrame:
    frame = pd.DataFrame(json.loads(path.read_text())["monitors"])
    frame["category"] = np.where(frame["kind"].eq("gleipnir"), "Gleipnir", "Other")
    return frame


def frontiers(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Each regime's frontier among the list-priced rows and its own Gleipnir rows."""
    frame = frame.copy()
    for regime, excluded in (("on_demand", "committed"), ("committed", "on_demand")):
        rows = frame["pricing"].ne(excluded)
        mask = pd.Series(False, index=frame.index)
        mask[rows] = pareto_frontier_mask(
            frame.loc[rows, "cost_per_1k"].to_numpy(),
            frame.loc[rows, metric].to_numpy(),
            cost_tolerance=FRONTIER_COST_TOLERANCE,
        )
        frame[f"{regime}_frontier"] = mask
    frame["frontier"] = frame["on_demand_frontier"].where(
        frame["pricing"].ne("committed"), frame["committed_frontier"]
    )
    return frame


def _label(row: pd.Series) -> str:
    name = str(row["monitor"]).replace("pplx-decider", "Decider")
    if name == "Kimi K3":
        name = "Kimi K3 (teacher)"
    return name + ("*" if row["pricing"] == "committed" else "")


def _distillation_arrow(axis: plt.Axes, base: pd.Series, target: pd.Series, y: str):
    start = (base["cost_per_1k"], base[y])
    end = (target["cost_per_1k"], target[y])
    axis.annotate(
        "",
        xy=end,
        xytext=start,
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


def plot(frame: pd.DataFrame, metric: str) -> plt.Figure:
    ylabel, _, ylim = METRICS[metric]
    frame = frontiers(frame, metric)
    set_plot_style()
    figure, axis = plt.subplots(figsize=(10.5, 6.6))
    axis.set_xscale("log")
    axis.set_xticks([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_xlim(frame["cost_per_1k"].min() / 1.35, frame["cost_per_1k"].max() * 1.3)
    axis.set_ylim(*ylim)

    for regime, style in (("on_demand", "-"), ("committed", (0, (5, 3)))):
        line = frame.loc[frame[f"{regime}_frontier"]].sort_values("cost_per_1k")
        axis.plot(
            line["cost_per_1k"],
            line[metric],
            color="#374151",
            linestyle=style,
            linewidth=2.3 if regime == "on_demand" else 2.0,
            alpha=0.82 if regime == "on_demand" else 0.7,
            zorder=1,
        )
    for base_name, target_name in BASE_TO_GLEIPNIR.items():
        base = frame.loc[frame["monitor"].eq(base_name)].iloc[0]
        target = frame.loc[
            frame["monitor"].eq(target_name) & frame["pricing"].eq("on_demand")
        ].iloc[0]
        _distillation_arrow(axis, base, target, metric)

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
        offset, ha, va, *coords = LABELS[metric].get(
            (row["monitor"], row["pricing"]), ((10, 0), "left", "center")
        )
        # A label placed in data coordinates gets a leader line to its point.
        leader = {"arrowstyle": "-", "color": "#9CA3AF", "linewidth": 0.7}
        axis.annotate(
            _label(row),
            (row["cost_per_1k"], row[metric]),
            xytext=offset,
            textcoords=coords[0] if coords else "offset points",
            arrowprops=leader if coords else None,
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
    ] + [
        Line2D([], [], color="#374151", linewidth=2.5, label="On-demand rate"),
        Line2D(
            [],
            [],
            color="#374151",
            linestyle=(0, (5, 3)),
            linewidth=2.2,
            label="Committed-use rate*",
        ),
    ]
    axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        prop={"size": 13, "weight": "bold"},
        handlelength=2.2,
        columnspacing=1.4,
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
