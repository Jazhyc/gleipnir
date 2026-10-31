"""Slide figures: SleightBench cost frontiers for decision monitors.

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

ROOT = Path(__file__).resolve().parents[1]
INK = "#1F2937"
MUTED = "#6B7280"
OURS = "#E45756"
OTHER = "#4C78A8"

# name: (input $/M, catch @1% FPR %, pAUROC@20, ours)
MONITORS = {
    "GPT-6.1 Terra": (1.00, 55.0, 0.734, False),
    "GPT-6 Sol": (2.00, 15.0, 0.515, False),
    "Kimi K3": (2.55, 0.0, 0.402, False),
    "pplx-decider v1.2": (0.02, 12.5, 0.300, False),
    "Gleipnir 27B": (0.0177, 20.0, 0.299, True),
    "GPT-6 Luna": (0.10, 5.0, 0.277, False),
    "Jev 1.13": (0.042, 12.5, 0.255, False),
    "Gleipnir 4B": (0.0042, 7.5, 0.209, True),
}
# label offsets in points, per metric
OFFSETS = {
    "catch": {"GPT-6.1 Terra": (-10, 0, "right"), "GPT-6 Sol": (8, 4, "left"),
              "Kimi K3": (-8, 6, "right"), "pplx-decider v1.2": (6, -14, "left"),
              "Gleipnir 27B": (-8, 6, "right"), "GPT-6 Luna": (8, -4, "left"),
              "Jev 1.13": (8, 4, "left"), "Gleipnir 4B": (8, -12, "left")},
    "pauroc": {"GPT-6.1 Terra": (-10, 0, "right"), "GPT-6 Sol": (0, 14, "center"),
               "Kimi K3": (0, -16, "center"), "pplx-decider v1.2": (6, 7, "left"),
               "Gleipnir 27B": (-8, 6, "right"), "GPT-6 Luna": (8, 2, "left"),
               "Jev 1.13": (6, -14, "left"), "Gleipnir 4B": (8, -12, "left")},
}


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


def frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    best, out = float("-inf"), []
    for cost, score in sorted(points):
        if score > best:
            out.append((cost, score))
            best = score
    return out


def plot_metric(metric: str, output_dir: Path) -> None:
    col = 1 if metric == "catch" else 2
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    pts = [(v[0], v[col]) for v in MONITORS.values()]
    front = frontier(pts)
    fx = [p[0] for p in front] + [4.0]
    fy = [p[1] for p in front] + [front[-1][1]]
    ax.step(fx, fy, where="post", color=MUTED, lw=1.4, ls="--", zorder=1)
    for name, vals in MONITORS.items():
        cost, score, ours = vals[0], vals[col], vals[3]
        color = OURS if ours else OTHER
        ax.scatter([cost], [score], s=90 if ours else 60, color=color, zorder=3,
                   edgecolor="white", lw=1.2)
        dx, dy, ha = OFFSETS[metric][name]
        label = name + ("*" if metric == "catch" and name == "Kimi K3" else "")
        ax.annotate(label, (cost, score), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=11.5, color=color,
                    fontweight="bold" if ours else "normal")
    ax.set_xscale("log")
    ax.set_xlim(0.002, 4.0)
    ax.set_xticks([0.003, 0.01, 0.03, 0.1, 0.3, 1, 3])
    ax.set_xticklabels(["$0.003", "$0.01", "$0.03", "$0.10", "$0.30", "$1", "$3"])
    ax.minorticks_off()
    ax.set_xlabel("Input price, $ per million tokens (log)")
    if metric == "catch":
        ax.set_ylim(-3, 62)
        ax.set_yticks([0, 20, 40, 60])
        ax.set_yticklabels(["0%", "20%", "40%", "60%"])
        ax.set_ylabel("Catch @1% FPR")
    else:
        ax.set_ylim(0.1, 0.8)
        ax.set_ylabel("pAUROC@20 (chance = 0.1)")
    ax.text(3.6, fy[-1], "frontier", color=MUTED, fontsize=10, ha="right",
            va="bottom")
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
    style()
    for metric in ("catch", "pauroc"):
        plot_metric(metric, args.output_dir)


if __name__ == "__main__":
    main()
