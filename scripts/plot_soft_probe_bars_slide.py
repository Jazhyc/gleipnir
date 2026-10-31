"""Slide figure: soft (Kimi K3) targets vs hard labels, probes and LoRA.

OOD pAUROC@20. Each probe bar is the ID-selected candidate for its model and
objective: probe design, checkpoint epoch and input (with or without the
monitor prompt) chosen on the seed-median ID score; the value is the
seed-median OOD over 5 seeds (probe_generalization full01/prompted01 and
probe_scale_27b records). The 4B LoRA reference is the 3-seed mean from
experiments/hard_label_seeds. The 27B LoRA reference is Gleipnir 27B (one
seed, teacher prompt, BF16; probe_scale_27b record); no hard-label 27B LoRA
was trained. The slide title carries the message.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INK = "#1F2937"
HARD = "#B8BEC6"
SOFT = "#8E6CC0"  # probing-section bar colour in the deck

GROUPS = [  # (label, hard, soft)
    ("Qwen3.5-4B\nhalf-depth probe", 0.7092, 0.7643),
    ("Qwen3.8-27B\nhalf-depth probe", 0.8406, 0.8481),
    ("Qwen3.5-4B\nLoRA, full model", 0.7164, 0.7921),
    ("Qwen3.8-27B\nLoRA, full model\n(1 seed)", None, 0.9352),
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
    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    width = 0.36
    centers = [0.0, 1.0, 2.25, 3.25]  # extra gap before the LoRA reference
    for i, (c, (_, hard, soft)) in enumerate(zip(centers, GROUPS, strict=True)):
        if hard is None:
            ax.text(c - width / 2, 0.2, "no hard-\nlabel run", ha="center",
                    va="bottom", color="#6B7280", fontsize=10)
        else:
            ax.bar(c - width / 2, hard, width, color=HARD,
                   label="Hard labels" if i == 0 else None)
        ax.bar(c + width / 2, soft, width, color=SOFT,
               label="Kimi K3 soft targets" if i == 0 else None)
        for x, v in ((c - width / 2, hard), (c + width / 2, soft)):
            if v is not None:
                ax.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                        color=INK, fontsize=12, fontweight="bold")
        if hard is not None:
            gain = round(soft, 2) - round(hard, 2)  # matches the printed values
            ax.text(c, max(hard, soft) + 0.11, f"+{gain:.2f}", ha="center",
                    va="bottom", color=SOFT, fontsize=14, fontweight="bold")
    ax.axvline(1.625, color="#C7CBD1", lw=1, ls=":")
    ax.axhline(0.1, color=INK, lw=1.8, ls=(0, (5, 3)), zorder=3)
    ax.text(3.25 + 0.5, 0.1, "chance\n(0.10)", color=INK, fontsize=12,
            ha="left", va="center", clip_on=False)
    ax.set_xlim(-0.5, 3.7)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("OOD pAUROC@20")
    ax.set_xticks(centers)
    ax.set_xticklabels([g[0] for g in GROUPS], fontsize=12)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper left", frameon=False, fontsize=12, ncol=2,
              bbox_to_anchor=(0.0, 1.12))
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(output_dir / f"soft_probe_bars_ood.{ext}", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures" / "slides")
    plot(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
