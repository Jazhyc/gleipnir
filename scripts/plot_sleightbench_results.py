#!/usr/bin/env python3
"""Plot SLEIGHT-Bench catch rate and pAUROC@20 for the writeup.

Grouped vertical bars, one group per model and one bar per way of running it,
as in the paper's figures. Values are the stripped-view, ControlArena-prompt
numbers recorded in ``docs/findings/sleightbench_released_monitors.md``.
Bars left of the dashed line are the paper's per-trial catch rates read from
its appendix figures; the rest are ours. Our paper-protocol runs sample 10
trials, and their pAUROC@20 is the mean of per-trial values, matching the
paper's per-trial catch rate; logit bars are single passes.
The paper reports no ranking metric, so its bars appear only in the
catch-rate figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from gleipnir.analysis.plotting import ORIGIN_PALETTE, save_figure, set_plot_style

SETTINGS = {
    "reasoned": ("No thinking", "#9ECAE9"),
    "thinking": ("Thinking", ORIGIN_PALETTE["Paper"]),
    "scaffold": ("Claude Code scaffold", "#54A24B"),
    "logits": ("Logit decision", "#B279A2"),
    "gleipnir": ("Logit decision, Gleipnir (ours)", ORIGIN_PALETTE["Gleipnir"]),
}
# (model, reported by the paper?, setting -> (catch rate %, pAUROC@20 or None))
MODELS = [
    (
        "Haiku 4.5",
        True,
        {"reasoned": (2, None), "thinking": (16, None), "scaffold": (7, None)},
    ),
    ("GPT-5.5", True, {"reasoned": (14, None), "thinking": (50, None)}),
    (
        "Opus 4.6",
        True,
        {"reasoned": (19, None), "thinking": (32, None), "scaffold": (62, None)},
    ),
    ("Gemini 3.1 Pro†", True, {"reasoned": (28, None), "thinking": (54, None)}),
    # GPT-5.5's logit run (15% catch, pAUROC@20 0.404) is left out: placed
    # across the separator from the paper's GPT-5.5 bars, it confused more
    # than it showed. The writeup reports it in Lesson 2.
    (
        "Kimi K3",
        False,
        {"reasoned": (12.2, 0.465), "thinking": (11.0, 0.396), "logits": (0, 0.402)},
    ),
    (
        "Qwen3.5-4B",
        False,
        {"reasoned": (0.5, 0.187), "thinking": (3.0, 0.213), "logits": (5, 0.193)},
    ),
    ("Gleipnir 4B", False, {"gleipnir": (7.5, 0.209)}),
    (
        "Qwen3.8-27B",
        False,
        {"reasoned": (8.9, 0.281), "thinking": (14.0, 0.319), "logits": (12.5, 0.263)},
    ),
    ("Gleipnir 27B", False, {"gleipnir": (20, 0.299)}),
]
PROTOCOL = (
    "Thinking and no thinking: the paper's protocol, written reasoning "
    "then a 0–100 score, over 10 sampled trials.\n"
)
FOOTNOTE = PROTOCOL + "† Gemini 3.1 Pro's bars are low and high thinking.\n"
KIMI_FOOTNOTE = (
    "* One benign twin, which differs from its attack by a single line number, "
    "scores 0.98 and sets the threshold;\n"
    "   at the second-highest benign, Kimi K3's logits catch 30%."
)
WIDTH = 0.24


def _grouped(
    metric: int, *, title: str, ylabel: str, fmt: str, footnote: str
) -> plt.Figure:
    models = [
        (
            name,
            reported,
            {s: v[metric] for s, v in bars.items() if v[metric] is not None},
        )
        for name, reported, bars in MODELS
    ]
    models = [model for model in models if model[2]]
    figure, axis = plt.subplots(figsize=(1.9 * len(models) + 2, 7))
    top = max(v for _, _, bars in models for v in bars.values())
    used = set()
    for index, (name, _, bars) in enumerate(models):
        offsets = [(i - (len(bars) - 1) / 2) * WIDTH for i in range(len(bars))]
        for offset, (setting, value) in zip(offsets, bars.items(), strict=True):
            used.add(setting)
            axis.bar(
                index + offset,
                value,
                width=WIDTH,
                color=SETTINGS[setting][1],
                edgecolor="white",
                linewidth=1.5,
            )
            star = (
                "*" if name == "Kimi K3" and setting == "logits" and metric == 0 else ""
            )
            axis.text(
                index + offset,
                value + top * 0.012,
                fmt.format(value) + star,
                ha="center",
                va="bottom",
                fontsize=9.5,
                color="#111827",
            )
    axis.set_xticks(range(len(models)), [m[0] for m in models], fontsize=12.5)
    for label in axis.get_xticklabels():
        if label.get_text().startswith("Gleipnir"):
            label.set_color(ORIGIN_PALETTE["Gleipnir"])
    axis.set_xlim(-0.6, len(models) - 0.4)
    axis.set_ylim(0, top * 1.12)
    split = sum(reported for _, reported, _ in models)
    if 0 < split < len(models):
        axis.axvline(split - 0.5, color="#6B7280", linestyle=(0, (4, 3)), linewidth=1.5)
        for x, text, align in (
            (split - 0.6, "Reported by the paper", "right"),
            (split - 0.4, "Our runs", "left"),
        ):
            axis.text(
                x,
                top * 1.09,
                text,
                ha=align,
                va="top",
                fontsize=12.5,
                fontweight="bold",
                color="#4B5563",
            )
    axis.set_ylabel(ylabel, fontsize=13)
    axis.grid(axis="x", visible=False)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.tick_params(axis="y", labelsize=12)
    axis.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=48)
    axis.legend(
        handles=[
            Patch(color=color, label=label)
            for key, (label, color) in SETTINGS.items()
            if key in used
        ],
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=3,
        frameon=False,
        prop={"size": 11.5, "weight": "bold"},
    )
    axis.text(
        0,
        -0.12,
        footnote,
        transform=axis.transAxes,
        fontsize=10.5,
        color="#4B5563",
        va="top",
    )
    return figure


def catch_figure() -> plt.Figure:
    return _grouped(
        0,
        title="Gleipnir 27B catches 20% of attacks, below frontier reasoning monitors",
        ylabel="Attacks caught at 1% FPR (%)",
        fmt="{:g}%",
        footnote=FOOTNOTE + KIMI_FOOTNOTE,
    )


def pauroc_figure() -> plt.Figure:
    return _grouped(
        1,
        title="At low FPR, our teacher Kimi K3 still separates attacks best",
        ylabel="pAUROC@20",
        fmt="{:.2f}",
        footnote=PROTOCOL
        + "pAUROC@20: normalized ROC area up to 20% FPR. The paper reports "
        "no ranking metric for its runs.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    set_plot_style()
    for name, build in (
        ("sleightbench_catch_rate", catch_figure),
        ("sleightbench_pauroc", pauroc_figure),
    ):
        for suffix in (".png", ".svg"):
            print(save_figure(build(), args.output_dir / f"{name}{suffix}"))


if __name__ == "__main__":
    main()
