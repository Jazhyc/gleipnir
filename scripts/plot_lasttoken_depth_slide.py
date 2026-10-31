"""Slide figure: where the monitoring decision forms, by layer.

A hard-label logistic-regression probe on the last token of the prompted
input (the `Prediction:` position) is fit at every layer of Qwen3.5-4B and
Qwen3.8-27B. Each point is the best ID pAUROC@20 over the regularization
grid. The x-axis is relative depth so both models share one scale; the
markers show the mid-depth layers the sequence probes use (15 of 32 and
31 of 64). The slide title carries the message, so the figure has none.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "probe_generalization"
INK = "#1F2937"
MUTED = "#6B7280"
SERIES = {
    "Qwen3.5-4B": ("full01/lasttoken_prompted.json", 15, "#4C78A8"),
    "Qwen3.8-27B": ("scale27b_prompted01/lasttoken_prompted.json", 31, "#E45756"),
}


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


def best_by_layer(path: Path, objective: str = "hard") -> dict[int, float]:
    best: dict[int, float] = {}
    for probe in json.loads(path.read_text())["probes"]:
        if probe["objective"] != objective:
            continue
        layer, score = probe["layer"], probe["id"]["pauroc_at_20"]
        best[layer] = max(best.get(layer, 0.0), score)
    return best


def plot(output_dir: Path) -> None:
    style()
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for name, (rel, chosen, color) in SERIES.items():
        best = best_by_layer(RESULTS / rel)
        n = len(best)
        layers = sorted(best)
        x = [(layer + 1) / n for layer in layers]
        y = [best[layer] for layer in layers]
        ax.plot(x, y, color=color, lw=2.2, label=f"{name} ({n} layers)")
        cx, cy = (chosen + 1) / n, best[chosen]
        ax.scatter([cx], [cy], s=90, color=color, edgecolor="white", lw=1.5, zorder=3)
        ax.annotate(f"layer {chosen}: {cy:.2f}", xy=(cx, cy),
                    xytext=(cx + 0.03, cy - 0.08), color=color, fontsize=12,
                    fontweight="bold")
    ax.axvspan(0.45, 0.52, color="#EEF1F4", zorder=0)
    ax.text(0.485, 0.22, "probe layer", color=MUTED, ha="center", fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.2, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["input", "25%", "50%", "75%", "output"])
    ax.set_xlabel("Relative depth")
    ax.set_ylabel("ID pAUROC@20")
    ax.legend(loc="upper left", frameon=False, title="Last-token probe",
              title_fontproperties={"weight": "bold"})
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png"):
        fig.savefig(output_dir / f"lasttoken_depth_id.{suffix}",
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures" / "slides")
    plot(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
