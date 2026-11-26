"""Figure fig:apps-probe (Appendix F.5): output of the model against a linear probe.

Plain matplotlib, no style package. Reads the "probe" entry of data.json and
writes held_out_probe.png here and held_out_probe.pdf to ../assets/.

    python make_probe_figure.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"

CONDITIONS = [
    ("none", "No text"),
    ("specific_accusation", "Specific\naccusation"),
    ("generic_accusation", "Generic\naccusation"),
]
MODELS = [("4b", "Gleipnir 4B"), ("base4b", "Qwen3.5-4B (unadapted)")]
SERIES = [
    ("output", "Output of the model", "#9C9A92"),
    ("probe", "Linear probe", "#6156D6"),
]
INK = "#3D3D3A"
MUTED = "#7A7972"


def main() -> None:
    data = json.loads((HERE / "data.json").read_text())["probe"]
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    x = np.arange(len(CONDITIONS))
    width = 0.32
    for ax, (model, title) in zip(axes, MODELS, strict=True):
        for i, (key, label, color) in enumerate(SERIES):
            entries = [data[model][c][key] for c, _ in CONDITIONS]
            v = np.array([e["value"] for e in entries])
            lo = np.array([e["ci95"][0] for e in entries])
            hi = np.array([e["ci95"][1] for e in entries])
            ax.bar(
                x + (i - 0.5) * (width + 0.02),
                v,
                width,
                color=color,
                label=label,
                zorder=2,
                yerr=(v - lo, hi - v),
                capsize=3,
                error_kw={"elinewidth": 1.1, "ecolor": "#1F1F1E"},
            )
        ax.axhline(0.5, color=MUTED, ls=":", lw=1.1, zorder=1)
        ax.set_title(title, color=MUTED, fontweight="bold", fontsize=11)
        ax.set_xticks(x, [lab for _, lab in CONDITIONS])
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 1.08)
        ax.set_xlim(-0.62, len(CONDITIONS) - 0.4)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", color="#E3E2DD", lw=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("AUROC,\nbackdoored vs benign")
    axes[1].tick_params(axis="y", left=False)
    axes[0].text(
        -0.58, 0.515, "chance", ha="left", va="bottom", fontsize=8, color=MUTED
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0),
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(HERE / "held_out_probe.png", dpi=200)
    fig.savefig(ASSETS / "held_out_probe.pdf")
    print("held_out_probe written")


if __name__ == "__main__":
    main()
