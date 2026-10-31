"""Slide figure: 27B last-token probe ID vs OOD by depth.

Prompted last-token logistic-regression probes with Kimi K3 soft targets.
Layers 24-44 come from the fixed-layer depth follow-up (C chosen on ID per
layer, OOD scored once); layer 50 is the ID-selected probe from the main
run. The 4B point is its ID-selected last-token probe (layer 17 of 32),
the only 4B layer scored OOD. Relative depth = (layer + 1) / n_layers.
The slide title carries the message.
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
C27 = "#E45756"
C4 = "#4C78A8"


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


def load() -> tuple[list[tuple[int, float, float]], tuple[int, float, float]]:
    depth = json.loads((RESULTS / "scale27b_prompted01/lasttoken_depth_prompted.json")
                       .read_text())
    pts = [(p["layer"], p["id"]["pauroc_at_20"], p["ood"]["pauroc_at_20"])
           for p in depth["probes"] if p["objective"] == "soft"]
    sel = json.loads((RESULTS / "scale27b_prompted01/ood_prompted.json").read_text())
    s = sel["last_token"]["soft"]
    pts.append((s["layer"], s["id"]["pauroc_at_20"], s["ood"]["pauroc_at_20"]))
    s4 = json.loads((RESULTS / "full01/ood_prompted.json").read_text())
    s4 = s4["last_token"]["soft"]
    p4 = (s4["layer"], s4["id"]["pauroc_at_20"], s4["ood"]["pauroc_at_20"])
    return sorted(pts), p4


def plot(output_dir: Path) -> None:
    style()
    pts, p4 = load()
    x = [(layer + 1) / 64 for layer, _, _ in pts]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ax.axvspan(0.45, 0.52, color="#EEF1F4", zorder=0)
    ax.text(0.485, 0.17, "half-depth\nprobes", color=MUTED, ha="center", fontsize=11)
    ax.plot(x, [p[1] for p in pts], color=C27, lw=2, ls="--", marker="o",
            ms=5, alpha=0.6, label="Qwen3.8-27B, ID")
    ax.plot(x, [p[2] for p in pts], color=C27, lw=2.6, marker="o", ms=6,
            label="Qwen3.8-27B, OOD")
    x4 = (p4[0] + 1) / 32
    ax.scatter([x4], [p4[2]], s=110, marker="D", color=C4, zorder=4,
               label=f"Qwen3.5-4B, OOD (layer {p4[0]}, only layer scored)")
    ax.annotate(f"{pts[-1][2]:.2f}", xy=(x[-1], pts[-1][2]), xytext=(6, -16),
                textcoords="offset points", color=C27, fontweight="bold")
    ax.annotate(f"{p4[2]:.2f}", xy=(x4, p4[2]), xytext=(9, -5),
                textcoords="offset points", color=C4, fontweight="bold")
    ax.set_xlim(0.3, 0.85)
    ax.set_ylim(0.1, 1.0)
    ax.set_xticks([0.375, 0.5, 0.625, 0.75])
    ax.set_xticklabels(["37.5%", "50%", "62.5%", "75%"])
    ax.set_xlabel("Relative depth")
    ax.set_ylabel("pAUROC@20")
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(output_dir / f"lasttoken_ood_depth.{ext}", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures" / "slides")
    plot(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
