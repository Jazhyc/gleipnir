"""Plot recall at 1% FPR against inference cost for the monitors we scored."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"  # portable SVG: text as outlines

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPOSITORY_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()
)
DEFAULT_INPUT = REPOSITORY_ROOT / "results/ood_recall_at_fpr/recall_at_fpr.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "figures/ood_recall_at_1pct_fpr_frontier.png"

# Gleipnir release models have two costs: committed-use and on-demand B200.
TWO_RATE = ("Gleipnir 4B", "Gleipnir 27B")
GROUPS = {
    "Unadapted, other output": ("Unadapted LLM, other output", "#9E9E9E"),
    "Unadapted, binary logits": ("Unadapted LLM, binary logits", "#4C78A8"),
    "Decision model": ("Decision model", "#F28E2B"),
    "Gleipnir": ("Gleipnir (ours)", "#E45756"),
}
LABEL_OFFSETS = {
    "Gleipnir 4B": (0, 10, "center"),
    "Gleipnir 27B": (0, 10, "center"),
    "Gleipnir 9B (earlier)": (7, -3, "left"),
    "d1": (7, -3, "left"),
    "d1-ultra": (-6, 6, "right"),
    "Mercury Decide": (-7, 1, "right"),
    "Microsoft-Decision-1": (7, -9, "left"),
    "Jev 1.13": (7, 0, "left"),
    "pplx-decider v1.2": (-7, -4, "right"),
    "Qwen3.5-4B": (7, -3, "left"),
    "Qwen3.5-9B": (-7, 4, "right"),
    "Qwen3.5-35B-A3B": (7, 3, "left"),
    "Qwen3.5-27B": (7, -3, "left"),
    "Qwen3.8-27B": (7, -3, "left"),
    "GPT-6 Luna": (-7, 4, "right"),
    "GPT-5.6 Luna": (7, -8, "left"),
    "Kimi K2.6": (0, -14, "center"),
    "Kimi K3": (0, 9, "center"),
    "Qwen3.5-9B, 0-10 score": (7, -3, "left"),
    "Qwen3.5-9B, reasoning + binary": (7, -10, "left"),
    "Qwen3.5-9B, reasoning + 0-10": (7, -3, "left"),
}
SHORT_LABELS = {
    "Qwen3.5-9B, 0-10 score": "9B, 0–10 score",
    "Qwen3.5-9B, reasoning + binary": "9B, reasoning + binary",
    "Qwen3.5-9B, reasoning + 0-10": "9B, reasoning + 0–10",
}


def frontier(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Nondominated points for lower cost and higher recall, ordered by cost."""
    kept: list[tuple[float, float, str]] = []
    best = float("-inf")
    for cost, recall, name in sorted(points, key=lambda point: (point[0], -point[1])):
        if recall > best:
            kept.append((cost, recall, name))
            best = recall
    return kept


def regime_points(
    monitors: list[dict], rate_index: int
) -> list[tuple[float, float, str]]:
    """Cost and recall per monitor; ``rate_index`` picks the Gleipnir GPU rate."""
    return [
        (
            m["cost_usd_per_1k"][rate_index if m["monitor"] in TWO_RATE else 0],
            m["mean_recall"],
            m["monitor"],
        )
        for m in monitors
    ]


def draw_frontier(
    ax: plt.Axes, points: list[tuple[float, float, str]], **style: object
) -> None:
    xs: list[float] = []
    ys: list[float] = []
    for index, (cost, recall, _) in enumerate(points):
        if index:
            xs.append(cost)
            ys.append(points[index - 1][1])
        xs.append(cost)
        ys.append(recall)
    xs.append(ax.get_xlim()[1])
    ys.append(points[-1][1])
    ax.plot(xs, ys, **style)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    monitors = json.loads(args.input.read_text())["monitors"]
    committed, on_demand = (
        frontier(regime_points(monitors, 0)),
        frontier(regime_points(monitors, 1)),
    )
    print("on-demand frontier:", ", ".join(name for _, _, name in on_demand))
    print("committed-use frontier:", ", ".join(name for _, _, name in committed))

    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=160)
    ax.set_xscale("log")
    ax.set_xlim(0.025, 45)
    ax.set_ylim(0.18, 0.93)
    for m in monitors:
        color = GROUPS[m["group"]][1]
        recall, cost = m["mean_recall"], m["cost_usd_per_1k"]
        if m["monitor"] in TWO_RATE:
            ax.plot(cost, [recall, recall], color=color, lw=1.2, alpha=0.6, zorder=2)
            ax.scatter(
                cost[1], recall, s=95, color=color, edgecolor="white", lw=0.8, zorder=4
            )
            ax.scatter(
                cost[0],
                recall,
                s=95,
                facecolor="white",
                edgecolor=color,
                lw=1.8,
                zorder=4,
            )
            anchor = float(np.sqrt(cost[0] * cost[1]))
        else:
            ax.scatter(
                cost[0], recall, s=55, color=color, edgecolor="white", lw=0.6, zorder=3
            )
            anchor = cost[0]
        dx, dy, align = LABEL_OFFSETS.get(m["monitor"], (7, -3, "left"))
        ax.annotate(
            SHORT_LABELS.get(m["monitor"], m["monitor"]),
            (anchor, recall),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=align,
            va="center",
            fontsize=8.2,
            color="#222222",
            fontweight="bold" if m["group"] == "Gleipnir" else None,
        )
    draw_frontier(ax, on_demand, color="#333333", lw=1.3, zorder=1)
    draw_frontier(ax, committed, color="#333333", lw=1.3, ls=(0, (4, 3)), zorder=1)

    ticks = [0.03, 0.1, 0.3, 1, 3, 10, 30]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["$0.03", "$0.10", "$0.30", "$1", "$3", "$10", "$30"])
    ax.set_xlabel("Inference cost per 1,000 evaluations (USD, log scale)")
    ax.set_ylabel("Recall at 1% FPR, mean of six OOD benchmarks")
    ax.grid(True, which="major", color="#E6E6E6", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    top = max(monitors, key=lambda m: m["mean_recall"])
    ax.set_title(
        f"At a 1% false-positive rate, {top['monitor']} catches about "
        f"{round(100 * top['mean_recall'])}% of harmful\n"
        "trajectories, the most of any monitor we scored",
        loc="left",
        fontsize=11.5,
        fontweight="bold",
    )
    types = ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                color=color,
                markeredgecolor="white",
                markersize=8,
                label=label,
            )
            for label, color in GROUPS.values()
        ],
        title="Monitor type",
        loc="lower right",
        bbox_to_anchor=(1.0, 0.20),
        fontsize=8.5,
        title_fontsize=9,
        edgecolor="#CCCCCC",
    )
    types._legend_box.align = "left"
    ax.add_artist(types)
    red = GROUPS["Gleipnir"][1]
    rates = ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                ls="-",
                color="#333333",
                markerfacecolor=red,
                markeredgecolor="white",
                markersize=8,
                label="On-demand B200",
            ),
            Line2D(
                [],
                [],
                marker="o",
                ls=(0, (4, 3)),
                color="#333333",
                markerfacecolor="white",
                markeredgecolor=red,
                markersize=8,
                label="Committed-use B200",
            ),
        ],
        title="Gleipnir GPU rate and frontier",
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        fontsize=8.5,
        title_fontsize=9,
        handlelength=3.2,
        edgecolor="#CCCCCC",
    )
    rates._legend_box.align = "left"
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in (args.output, args.output.with_suffix(".svg")):
        fig.savefig(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
