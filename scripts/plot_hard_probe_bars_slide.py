"""Slide figure: hard-label half-depth probes vs zero-shot and full Gleipnir.

OOD pAUROC@20 for Qwen3.5-4B (probe at layer 15 of 32) and Qwen3.8-27B
(layer 31 of 64). Probe values are the monitor-prompt Kramár sequence
probes with hard labels, median of 5 seeds; all rows are BF16. Values are
the ones recorded in the probe_generalization experiment records. The
slide title carries the message, so the figure has none.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INK = "#1F2937"
MUTED = "#6B7280"
GREY = "#B8BEC6"
DARK = "#4B5563"
PURPLE = "#8E6CC0"  # probing-section bar colour in the deck

ROWS = [  # (label, colour, 4B, 27B)
    ("Base model,\nzero-shot", GREY, 0.62, 0.89),
    ("Half-depth\nprobe", PURPLE, 0.71, 0.84),
    ("Gleipnir\n(full model)", DARK, 0.80, 0.94),
]
PANELS = [
    ("Qwen3.5-4B · probe at layer 15/32", 2),
    ("Qwen3.8-27B · probe at layer 31/64", 3),
]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Lato", "Cantarell", "DejaVu Sans"],
            "font.size": 13,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#C7CBD1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "savefig.dpi": 300,
            "svg.fonttype": "path",
        }
    )


def plot(output_dir: Path) -> None:
    style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, (title, col) in zip(axes, PANELS, strict=True):
        xs = range(len(ROWS))
        vals = [row[col] for row in ROWS]
        ax.bar(xs, vals, width=0.62, color=[row[1] for row in ROWS])
        for x, v in zip(xs, vals, strict=True):
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                    color=INK, fontsize=13, fontweight="bold")
        ax.axhline(0.1, color=INK, lw=1.8, ls=(0, (5, 3)), zorder=3)
        ax.set_xlim(-0.5, 2.5)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([row[0] for row in ROWS], fontsize=12)
        ax.set_title(title, color=INK, fontsize=13, loc="left")
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_ylabel("OOD pAUROC@20")
    axes[1].text(2.58, 0.1, "chance\n(0.10)", color=INK, fontsize=12,
                 ha="left", va="center", clip_on=False)
    fig.tight_layout(w_pad=2.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(output_dir / f"hard_probe_bars_ood.{ext}", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures" / "slides")
    plot(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
