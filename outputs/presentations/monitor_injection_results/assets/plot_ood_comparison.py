"""Regenerate the deck's charts from local aggregate inputs only.

Usage: python assets/plot_ood_comparison.py
Requires matplotlib; no project modules, datasets, or checkpoints are needed.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS = Path(__file__).resolve().parent
BLUE = "#2463A6"
PAPER = "#91ADC9"
NAVY = "#182B49"


def style_axes(ax: plt.Axes) -> None:
    """Keep the charts legible against the deck's white backgrounds."""
    ax.set_xlim(0, 1.13)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("pAUROC@20 (higher is better)")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#DDE5EE", linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#91ADC9")
    ax.tick_params(axis="y", length=0)


def main() -> None:
    """Render the aggregate and per-source comparisons as PNG figures."""
    data = json.loads((ASSETS / "ood_comparison.json").read_text())
    plt.rcParams.update(
        {
            "font.size": 12,
            "text.color": NAVY,
            "axes.labelcolor": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "savefig.dpi": 220,
        }
    )
    macro = data["macro"]
    fig, ax = plt.subplots(figsize=(6.3, 3.9), layout="constrained")
    values = [macro["paper_pauroc_at_20"], macro["ours_pauroc_at_20"]]
    ax.barh([1, 0], values, height=0.48, color=[PAPER, BLUE])
    ax.errorbar(
        values[0],
        1,
        xerr=macro["paper_reported_sd"],
        fmt="none",
        color=NAVY,
        capsize=5,
        linewidth=1.5,
    )
    ax.set_yticks([1, 0], ["Published\nSFT + RL", "Our logit\ndistillation"])
    for y, value in zip([1, 0], values, strict=True):
        ax.text(value + 0.06, y, f"{value:.4f}", va="center", fontweight="bold")
    ax.set_ylim(-0.65, 1.65)
    style_axes(ax)
    fig.savefig(ASSETS / "ood_mean.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.6, 3.9), layout="constrained")
    groups = data["sources"]
    positions = list(range(len(groups)))
    paper_values = [g["paper_pauroc_at_20"] for g in groups]
    ours_values = [g["ours_pauroc_at_20"] for g in groups]
    ax.barh(
        [p - 0.18 for p in positions],
        paper_values,
        height=0.31,
        color=PAPER,
        label="Published SFT + RL",
    )
    ax.barh(
        [p + 0.18 for p in positions],
        ours_values,
        height=0.31,
        color=BLUE,
        label="Our logit distillation",
    )
    ax.errorbar(
        paper_values,
        [p - 0.18 for p in positions],
        xerr=[g["paper_reported_sd"] for g in groups],
        fmt="none",
        color=NAVY,
        capsize=3,
        linewidth=1,
    )
    for position, paper_value, ours_value, group in zip(
        positions, paper_values, ours_values, groups, strict=True
    ):
        ax.text(
            paper_value + group["paper_reported_sd"] + 0.013,
            position - 0.18,
            f"{paper_value:.3f}",
            va="center",
            fontsize=10,
        )
        ax.text(
            ours_value + 0.013,
            position + 0.18,
            f"{ours_value:.3f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_yticks(positions, [f"{g['label']}  (n={g['n']:,})" for g in groups])
    ax.invert_yaxis()
    style_axes(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.savefig(ASSETS / "ood_by_source.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
