#!/usr/bin/env python3
"""Plot SLEIGHT-Bench catch rate and pAUROC@20 for the writeup.

Values are the stripped-view, ControlArena-prompt numbers recorded in
``docs/findings/sleightbench_released_monitors.md``: paper rows are per-trial
catch rates read from the paper's appendix figures; our paper-protocol rows
average 10 sampled trials; logit rows are single passes. The paper reports no
ranking metric, so its rows appear only in the catch-rate figure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from gleipnir.analysis.plotting import ORIGIN_PALETTE, save_figure, set_plot_style

PALETTE = {
    "paper": ORIGIN_PALETTE["Paper"],
    "reasoned": "#9ECAE9",
    "logits": "#B279A2",
    "gleipnir": ORIGIN_PALETTE["Gleipnir"],
}
GROUP_LABELS = {
    "paper": "Reasoned score (paper's runs)",
    "reasoned": "Reasoned score (our runs, paper protocol)",
    "logits": "Logit decision",
    "gleipnir": "Gleipnir (ours, logit decision)",
}


@dataclass(frozen=True)
class Row:
    label: str
    group: str
    catch: float
    pauroc: float | None = None
    note: str = ""


ROWS = [
    Row("Opus 4.6, Claude Code", "paper", 62),
    Row("Gemini 3.1 Pro, high thinking", "paper", 54),
    Row("GPT-5.5, thinking", "paper", 50),
    Row("Opus 4.6, thinking", "paper", 32),
    Row("Gemini 3.1 Pro, low thinking", "paper", 28),
    Row("Opus 4.6", "paper", 19),
    Row("GPT-5.5", "paper", 14),
    Row("Haiku 4.5", "paper", 2),
    Row("Qwen3.8-27B, thinking", "reasoned", 14.0, 0.359),
    Row("Kimi K3", "reasoned", 12.2, 0.523),
    Row("Kimi K3, thinking", "reasoned", 11.0, 0.444),
    Row("Qwen3.8-27B", "reasoned", 8.9, 0.318),
    Row("Qwen3.5-4B, thinking", "reasoned", 3.0, 0.237),
    Row("Qwen3.5-4B", "reasoned", 0.5, 0.213),
    Row("GPT-5.5", "logits", 15, 0.404),
    Row("Qwen3.8-27B", "logits", 12.5, 0.263),
    Row("Qwen3.5-4B", "logits", 5, 0.193),
    Row("Kimi K3", "logits", 0, 0.402, note="*"),
    Row("Gleipnir 27B", "gleipnir", 20, 0.299),
    Row("Gleipnir 4B", "gleipnir", 7.5, 0.209),
]
KIMI_FOOTNOTE = (
    "* One benign twin, which differs from its attack by a single line number, "
    "scores 0.98 and sets the threshold.\n"
    "   Thresholded at the second-highest benign instead, Kimi K3's logits "
    "catch 30%."
)


def _bars(
    rows: list[Row], values: list[float], *, title: str, xlabel: str, fmt: str
) -> plt.Figure:
    groups = list(dict.fromkeys(row.group for row in rows))
    positions, y = [], 0.0
    for index, row in enumerate(rows):
        if index and row.group != rows[index - 1].group:
            y += 0.7
        positions.append(y)
        y += 1
    figure, axis = plt.subplots(figsize=(11, 0.42 * y + 2.2))
    axis.barh(
        positions,
        values,
        height=0.72,
        color=[PALETTE[row.group] for row in rows],
        edgecolor="white",
        linewidth=2,
    )
    top = max(values)
    for position, value, row in zip(positions, values, rows, strict=True):
        axis.text(
            value + top * 0.012,
            position,
            fmt.format(value) + row.note,
            va="center",
            fontsize=12,
            color="#111827",
        )
    axis.set_yticks(positions, [row.label for row in rows], fontsize=12.5)
    axis.invert_yaxis()
    axis.set_xlim(0, top * 1.13)
    axis.set_xlabel(xlabel, fontsize=13)
    axis.grid(axis="y", visible=False)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.tick_params(axis="x", labelsize=12)
    axis.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=14)
    axis.legend(
        handles=[Patch(color=PALETTE[g], label=GROUP_LABELS[g]) for g in groups],
        loc="upper left",
        bbox_to_anchor=(-0.02, -0.1),
        ncol=2,
        frameon=False,
        prop={"size": 11.5, "weight": "bold"},
    )
    return figure


def catch_figure() -> plt.Figure:
    figure = _bars(
        ROWS,
        [row.catch for row in ROWS],
        title="Gleipnir 27B catches 20% of attacks, below frontier reasoning monitors",
        xlabel="Attacks caught at 1% FPR (%)",
        fmt="{:g}%",
    )
    figure.axes[0].text(
        -0.02,
        -0.2,
        KIMI_FOOTNOTE,
        transform=figure.axes[0].transAxes,
        fontsize=11,
        color="#4B5563",
        va="top",
    )
    return figure


def pauroc_figure() -> plt.Figure:
    groups = list(dict.fromkeys(row.group for row in ROWS))
    rows = sorted(
        (Row(r.label, r.group, r.catch, r.pauroc) for r in ROWS if r.pauroc),
        key=lambda r: (groups.index(r.group), -r.pauroc),
    )
    return _bars(
        rows,
        [row.pauroc for row in rows],
        title="At low FPR, our teacher Kimi K3 still separates attacks best",
        xlabel="pAUROC@20 (normalized ROC area up to 20% FPR)",
        fmt="{:.3f}",
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
            figure = build()
            print(save_figure(figure, args.output_dir / f"{name}{suffix}"))


if __name__ == "__main__":
    main()
