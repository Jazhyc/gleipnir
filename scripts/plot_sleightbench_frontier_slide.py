"""Slide figures: SleightBench cost frontiers for decision monitors.

Styled like the OOD frontier (scripts/plot_tool_trajectory_ood_frontier.py).
Stripped view, single pass, no reasoning (findings:
docs/findings/sleightbench_released_monitors.md, runs sb05-sb13). Price is
uncached input $/M: API list prices (Kimi K3 via OpenRouter), Gleipnir at
the committed-GPU ($3.44/h B200) break-even from
docs/research/tool_trajectory_inference_economics.md. Writes one figure per
metric: catch @1% FPR and pAUROC@20. Slide titles carry the messages.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter

from gleipnir.analysis.plotting import (
    ORIGIN_PALETTE,
    pareto_frontier_mask,
    set_plot_style,
)

ROOT = Path(__file__).resolve().parents[1]
DECISION = "#B279A2"
LABEL = "#4B5563"
LINE = "#374151"

# name: (input $/M, catch @1% FPR, pAUROC@20, ours)
MONITORS = {
    "GPT-6.1 Terra": (1.00, 0.55, 0.734, False),
    "GPT-6 Sol": (2.00, 0.15, 0.515, False),
    "Kimi K3 (teacher)": (2.55, 0.0, 0.402, False),
    "Decider v1.2": (0.02, 0.125, 0.300, False),
    "Gleipnir 27B": (0.0177, 0.20, 0.299, True),
    "GPT-6 Luna": (0.10, 0.05, 0.277, False),
    "Jev 1.13": (0.042, 0.125, 0.255, False),
    "Gleipnir 4B": (0.0042, 0.075, 0.209, True),
}
# label offsets in points (dx, dy, ha), per metric
OFFSETS = {
    "catch": {"GPT-6.1 Terra": (-12, 0, "right"), "GPT-6 Sol": (0, 15, "center"),
              "Kimi K3 (teacher)": (-12, 4, "right"), "Decider v1.2": (8, -15, "left"),
              "Gleipnir 27B": (-12, 8, "right"), "GPT-6 Luna": (10, -6, "left"),
              "Jev 1.13": (10, 5, "left"), "Gleipnir 4B": (0, -18, "center")},
    "pauroc": {"GPT-6.1 Terra": (-12, 0, "right"), "GPT-6 Sol": (0, 15, "center"),
               "Kimi K3 (teacher)": (0, -17, "center"), "Decider v1.2": (8, 10, "left"),
               "Gleipnir 27B": (-12, 8, "right"), "GPT-6 Luna": (10, 5, "left"),
               "Jev 1.13": (8, -15, "left"), "Gleipnir 4B": (0, -18, "center")},
}


def plot_metric(metric: str, output_dir: Path) -> None:
    col = 1 if metric == "catch" else 2
    names = list(MONITORS)
    cost = np.array([MONITORS[n][0] for n in names])
    score = np.array([MONITORS[n][col] for n in names])
    ours = np.array([MONITORS[n][3] for n in names])
    on_front = pareto_frontier_mask(cost, score)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(10.0, 6.4))
    ax.set_xscale("log")
    ax.set_xticks([0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlim(0.0028, 3.6)

    order = np.argsort(cost[on_front])
    ax.plot(cost[on_front][order], score[on_front][order], color=LINE, lw=2.3,
            alpha=0.82, zorder=1)
    for i, name in enumerate(names):
        color = ORIGIN_PALETTE["Gleipnir"] if ours[i] else DECISION
        marker = "D" if ours[i] else "s"
        big = on_front[i]
        ax.scatter(cost[i], score[i], s=165 if big else 92, marker=marker,
                   color=color, alpha=1.0 if big else 0.78,
                   edgecolor="#111827" if big else LINE,
                   linewidth=1.5 if big else 0.65, zorder=3 if big else 2)
        dx, dy, ha = OFFSETS[metric][name]
        label = name + ("*" if metric == "catch" and name.startswith("Kimi") else "")
        ax.annotate(label, (cost[i], score[i]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=13,
                    fontweight="bold", color="#111827" if big else LABEL)

    ax.annotate("Better", xy=(0.015, 0.975), xytext=(0.085, 0.86),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=13, fontweight="bold", color="#6B7280",
                ha="center", va="center",
                arrowprops={"arrowstyle": "-|>", "color": "#6B7280",
                            "linewidth": 1.8, "mutation_scale": 16,
                            "shrinkA": 12, "shrinkB": 0})
    if metric == "catch":
        ax.set_ylim(-0.04, 0.66)
        ax.set_ylabel("Catch rate at 1% FPR")
    else:
        ax.set_ylim(0.15, 0.8)
        ax.set_ylabel("Detection performance\n(pAUROC@20)")
    ax.set_xlabel("Input price (USD per million tokens; log scale)")

    handles = [
        Line2D([], [], marker="s", ls="", color=DECISION, markeredgecolor=LINE,
               markersize=10, label="Decision"),
        Line2D([], [], marker="D", ls="", color=ORIGIN_PALETTE["Gleipnir"],
               markeredgecolor=LINE, markersize=10, label="Gleipnir (ours)"),
        Line2D([], [], color=LINE, lw=2.3, label="Frontier"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=13, title="Monitor type",
              title_fontsize=13, frameon=True, bbox_to_anchor=(0.14, 0.99))
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(output_dir / f"sleightbench_frontier_{metric}.{ext}",
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures" / "slides")
    args = parser.parse_args()
    for metric in ("catch", "pauroc"):
        plot_metric(metric, args.output_dir)


if __name__ == "__main__":
    main()
