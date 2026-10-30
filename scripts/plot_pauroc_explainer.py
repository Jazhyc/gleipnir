"""Slide figure explaining pAUROC@20 on an illustrative ROC curve.

The curve is schematic (TPR = FPR ** 0.25), not a measured monitor. The
shaded area under it up to 20% FPR, divided by 0.2, is the pAUROC@20; the
chance diagonal gives 0.1 and a perfect monitor 1.0. The slide title carries
the message, so the figure has none.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INK = "#1F2937"
MUTED = "#6B7280"
ACCENT = "#E45756"
IGNORED = "#EEF1F4"
CUTOFF = 0.2
EXPONENT = 0.25


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Lato", "Cantarell", "DejaVu Sans"],
            "font.size": 13,
            "axes.labelsize": 13,
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
    fpr = np.linspace(0, 1, 501)
    tpr = fpr**EXPONENT
    pauroc = CUTOFF**EXPONENT / (1 + EXPONENT)  # integral to the cutoff / 0.2

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axvspan(CUTOFF, 1, color=IGNORED, zorder=0)
    inside = fpr <= CUTOFF
    ax.fill_between(fpr[inside], 0, tpr[inside], color=ACCENT, alpha=0.25, lw=0)
    ax.plot(fpr, tpr, color=ACCENT, lw=2.2)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.5, ls="--")
    ax.axvline(CUTOFF, color=INK, lw=1.2)

    ax.text(0.68, 0.07, "not counted: too many false alarms", color=MUTED,
            ha="center", va="center", fontsize=12)
    ax.annotate(f"pAUROC@20 = area ÷ 0.2\n= {pauroc:.2f}", xy=(0.12, 0.25),
                xytext=(0.36, 0.24), color=INK, ha="left", va="center",
                fontsize=13, fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.2})
    ax.text(0.5, 0.92, "example monitor", color=ACCENT, ha="center",
            fontsize=12, fontweight="bold")
    ax.text(0.82, 0.72, "chance = 0.1", color=MUTED, ha="center",
            va="center", rotation=31, rotation_mode="anchor", fontsize=12)
    ax.text(CUTOFF + 0.01, 1.03, "20% FPR", color=INK, fontsize=12,
            fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("False-positive rate (benign runs flagged)")
    ax.set_ylabel("True-positive rate\n(attacks caught)")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png"):
        fig.savefig(output_dir / f"pauroc20_explainer.{suffix}",
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures" / "slides")
    plot(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
