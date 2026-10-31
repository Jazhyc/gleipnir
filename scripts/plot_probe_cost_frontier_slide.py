"""Slide figure: OOD cost frontier with half-depth probes.

Styled like the OOD frontier (scripts/plot_tool_trajectory_ood_frontier.py).
y: mean OOD pAUROC@20 over the six OOD datasets. x: USD per 1,000 OOD
evaluations, as in the main frontier. Gleipnir, decision-model and base-model
rows come from docs/research/tool_trajectory_ood_frontier.md (Gleipnir at the
committed-GPU B200 break-even, $3.44/h; decision models at API list price;
base models at the hosted-price proxy used there). Half-depth probe costs
scale the full model's committed cost by the measured rows/s ratio on one
B200 (experiments/probe_optimized_serving: 4B 26.4 -> 48.5 rows/s, 27B
prompted 5.68 -> 11.07 rows/s). Probes are the deployed soft-target attention
probes (OOD 0.750 and 0.849 on the optimized stack).
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
PROBE = "#8E6CC0"  # probing-section colour in the deck
LABEL = "#4B5563"
LINE = "#374151"
BASE = "#9CA3AF"
G4, G27 = 0.0366, 0.1677  # committed $/1K, frontier registry


# name: ($/1K evaluations, OOD pAUROC@20, kind, label offset (dx, dy, ha))
MONITORS = {
    "Gleipnir 4B": (G4, 0.7703, "full", (10, -14, "left")),
    "4B probe": (G4 * 26.4 / 48.5, 0.750, "probe", (0, -20, "center")),
    "Gleipnir 27B": (G27, 0.9508, "full", (-12, 10, "right")),
    "27B probe": (G27 * 5.68 / 11.07, 0.849, "probe", (-12, 8, "right")),
    "Decider v1.2": (0.1763, 0.8956, "decision", (10, -6, "left")),
    "Jev 1.13": (0.3887, 0.8495, "decision", (0, 18, "center")),
    "GPT-6 Luna": (0.8355, 0.8365, "decision", (10, -14, "left")),
    "Qwen3.5-4B base": (0.2842, 0.6175, "base", (-12, 0, "right")),
    "Qwen3.8-27B base": (1.8476, 0.8901, "base", (0, 18, "center")),
}


def plot(output_dir: Path) -> None:
    names = list(MONITORS)
    cost = np.array([MONITORS[n][0] for n in names])
    score = np.array([MONITORS[n][1] for n in names])
    on_front = pareto_frontier_mask(cost, score)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(10.0, 6.4))
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlim(0.012, 3.2)
    ax.set_ylim(0.58, 0.98)

    order = np.argsort(cost[on_front])
    ax.plot(cost[on_front][order], score[on_front][order], color=LINE, lw=2.3,
            alpha=0.82, zorder=1)
    style = {"full": (ORIGIN_PALETTE["Gleipnir"], "D"), "probe": (PROBE, "D"),
             "decision": (DECISION, "s"), "base": (BASE, "o")}
    for i, name in enumerate(names):
        c, kind = MONITORS[name][:2], MONITORS[name][2]
        color, marker = style[kind]
        big = on_front[i]
        ax.scatter(*c, s=165 if big else 92, marker=marker, color=color,
                   alpha=1.0 if big else 0.78,
                   edgecolor="#111827" if big else LINE,
                   linewidth=1.5 if big else 0.65, zorder=3 if big else 2)
        dx, dy, ha = MONITORS[name][3]
        ax.annotate(name, c, xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="center", fontsize=13, fontweight="bold",
                    color="#111827" if big else LABEL)

    ax.annotate("Better", xy=(0.015, 0.975), xytext=(0.085, 0.86),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=13, fontweight="bold", color="#6B7280",
                ha="center", va="center",
                arrowprops={"arrowstyle": "-|>", "color": "#6B7280",
                            "linewidth": 1.8, "mutation_scale": 16,
                            "shrinkA": 12, "shrinkB": 0})
    ax.set_ylabel("Detection performance\n(pAUROC@20, mean over OOD)")
    ax.set_xlabel("Inference cost (USD per 1,000 evaluations; log scale)")
    handles = [
        Line2D([], [], marker="s", ls="", color=DECISION, markeredgecolor=LINE,
               markersize=10, label="Decision"),
        Line2D([], [], marker="o", ls="", color=BASE, markeredgecolor=LINE,
               markersize=10, label="Base model, zero-shot"),
        Line2D([], [], marker="D", ls="", color=ORIGIN_PALETTE["Gleipnir"],
               markeredgecolor=LINE, markersize=10, label="Gleipnir, full model"),
        Line2D([], [], marker="D", ls="", color=PROBE, markeredgecolor=LINE,
               markersize=10, label="Gleipnir, half-depth probe"),
        Line2D([], [], color=LINE, lw=2.3, label="Frontier"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=13, title="Monitor type",
              title_fontsize=13, frameon=True)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(output_dir / f"probe_cost_frontier.{ext}", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures" / "slides")
    plot(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
