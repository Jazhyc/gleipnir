"""Plot the paper and Gleipnir tool-monitor distillation pipelines."""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gleipnir-matplotlib")
)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from gleipnir.plotting import ORIGIN_PALETTE, save_figure, set_plot_style

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "figures/distillation_pipeline_comparison.png"
DEFAULT_VECTOR_OUTPUT = REPOSITORY_ROOT / "figures/distillation_pipeline_comparison.svg"

PAPER_STAGES = (
    "Gemini 2.5 Pro samples\n4 structured rationales",
    "Verdict matches label?",
    "Incorrect candidate regenerated\nwith label (up to 3 retries)",
    "Claude Sonnet 4.5 scores candidates\nkeep best if quality ≥ 7/10",
    "Student is fine-tuned on selected rationale\n+ 0–10 severity score",
    "Optional GRPO score refinement",
)
GLEIPNIR_STAGES = (
    "Kimi K3 provides one soft target\np(0),  p(1)\n0 = benign  ·  1 = misaligned",
    "Student matches the teacher probabilities\nat the 0/1 decision boundary",
)
GROUND_TRUTH_DISCLOSURE = "shown to Gemini\nonly on retry"

INK = "#172033"
MUTED_INK = "#526174"
NEUTRAL_FILL = "#F4F6F8"
NEUTRAL_EDGE = "#8996A8"
PAPER_COLOR = ORIGIN_PALETTE["Paper"]
GLEIPNIR_COLOR = ORIGIN_PALETTE["Gleipnir"]
PAPER_FILL = "#EAF2FA"
GLEIPNIR_FILL = "#FCEDEC"
LABEL_FILL = "#FFF4D6"
LABEL_EDGE = "#C78A16"


@dataclass(frozen=True)
class Box:
    """Axis-relative bounds for one rounded pipeline box."""

    center_x: float
    center_y: float
    width: float
    height: float

    @property
    def left(self) -> float:
        return self.center_x - self.width / 2

    @property
    def right(self) -> float:
        return self.center_x + self.width / 2

    @property
    def top(self) -> float:
        return self.center_y + self.height / 2

    @property
    def bottom(self) -> float:
        return self.center_y - self.height / 2


def _rounded_box(
    axis: Axes,
    box: Box,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 12.0,
    linewidth: float = 1.7,
    linestyle: str = "-",
    text_color: str = INK,
    zorder: int = 3,
) -> None:
    patch = FancyBboxPatch(
        (box.left, box.bottom),
        box.width,
        box.height,
        boxstyle="round,pad=0.009,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        transform=axis.transAxes,
        clip_on=False,
        zorder=zorder,
    )
    axis.add_patch(patch)
    axis.text(
        box.center_x,
        box.center_y,
        text,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=text_color,
        linespacing=1.22,
        zorder=zorder + 1,
    )


def _arrow(
    axis: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED_INK,
    connectionstyle: str = "arc3,rad=0",
    linestyle: str = "-",
    linewidth: float = 1.8,
    zorder: int = 2,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        transform=axis.transAxes,
        clip_on=False,
        zorder=zorder,
    )
    axis.add_patch(patch)


def _vertical_arrow(axis: Axes, source: Box, target: Box, **kwargs: object) -> None:
    _arrow(
        axis,
        (source.center_x, source.bottom - 0.003),
        (target.center_x, target.top + 0.003),
        **kwargs,
    )


def _panel_background(
    axis: Axes,
    *,
    x: float,
    width: float,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        (x, 0.035),
        width,
        0.89,
        boxstyle="round,pad=0.008,rounding_size=0.025",
        facecolor=color,
        edgecolor="none",
        alpha=0.055,
        transform=axis.transAxes,
        clip_on=False,
        zorder=0,
    )
    axis.add_patch(patch)


def _panel_heading(
    axis: Axes,
    *,
    center_x: float,
    label: str,
    title: str,
    subtitle: str,
    color: str,
) -> None:
    axis.text(
        center_x,
        0.91,
        f"{label}  {title}",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=16.5,
        fontweight="bold",
        color=color,
    )
    axis.text(
        center_x,
        0.875,
        subtitle,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="bold",
        color=MUTED_INK,
    )


def _draw_paper_pipeline(axis: Axes) -> None:
    input_box = Box(0.255, 0.80, 0.32, 0.065)
    teacher_box = Box(0.21, 0.685, 0.28, 0.085)
    check_box = Box(0.155, 0.555, 0.17, 0.055)
    retry_box = Box(0.375, 0.555, 0.18, 0.075)
    label_box = Box(0.43, 0.685, 0.085, 0.068)
    judge_box = Box(0.255, 0.405, 0.37, 0.092)
    student_box = Box(0.255, 0.255, 0.35, 0.082)
    rl_box = Box(0.255, 0.145, 0.22, 0.042)
    monitor_box = Box(0.255, 0.065, 0.30, 0.055)

    _rounded_box(
        axis,
        input_box,
        "Action-only trajectory",
        facecolor=NEUTRAL_FILL,
        edgecolor=NEUTRAL_EDGE,
    )
    _rounded_box(
        axis,
        teacher_box,
        PAPER_STAGES[0],
        facecolor=PAPER_FILL,
        edgecolor=PAPER_COLOR,
    )
    _rounded_box(
        axis,
        check_box,
        PAPER_STAGES[1],
        facecolor="white",
        edgecolor=PAPER_COLOR,
        fontsize=10.3,
    )
    _rounded_box(
        axis,
        retry_box,
        PAPER_STAGES[2],
        facecolor=PAPER_FILL,
        edgecolor=PAPER_COLOR,
        fontsize=9.9,
    )
    _rounded_box(
        axis,
        label_box,
        f"Ground-truth class\n{GROUND_TRUTH_DISCLOSURE}",
        facecolor=LABEL_FILL,
        edgecolor=LABEL_EDGE,
        fontsize=7.5,
        linewidth=1.4,
    )
    _rounded_box(
        axis,
        judge_box,
        PAPER_STAGES[3],
        facecolor=PAPER_FILL,
        edgecolor=PAPER_COLOR,
        fontsize=10.9,
    )
    _rounded_box(
        axis,
        student_box,
        PAPER_STAGES[4],
        facecolor=PAPER_FILL,
        edgecolor=PAPER_COLOR,
        fontsize=11.0,
    )
    _rounded_box(
        axis,
        rl_box,
        PAPER_STAGES[5],
        facecolor="white",
        edgecolor=PAPER_COLOR,
        fontsize=9.8,
        linestyle="--",
        linewidth=1.4,
    )
    _rounded_box(
        axis,
        monitor_box,
        "Deployable action-only monitor",
        facecolor=NEUTRAL_FILL,
        edgecolor=INK,
        fontsize=11.1,
    )

    _vertical_arrow(axis, input_box, teacher_box)
    _arrow(
        axis,
        (teacher_box.center_x, teacher_box.bottom - 0.003),
        (check_box.center_x, check_box.top + 0.003),
        connectionstyle="arc3,rad=0.12",
    )
    _arrow(
        axis,
        (check_box.right + 0.003, check_box.center_y),
        (retry_box.left - 0.003, retry_box.center_y),
        color=PAPER_COLOR,
    )
    axis.text(
        0.265,
        0.57,
        "incorrect",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=PAPER_COLOR,
    )
    _arrow(
        axis,
        (label_box.center_x, label_box.bottom - 0.003),
        (retry_box.center_x + 0.055, retry_box.top + 0.003),
        color=LABEL_EDGE,
        connectionstyle="arc3,rad=0.16",
        linestyle="--",
        linewidth=1.5,
    )
    _arrow(
        axis,
        (check_box.center_x, check_box.bottom - 0.003),
        (judge_box.center_x - 0.07, judge_box.top + 0.003),
        connectionstyle="arc3,rad=-0.12",
    )
    axis.text(
        0.145,
        0.485,
        "correct",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color=MUTED_INK,
    )
    _arrow(
        axis,
        (retry_box.center_x, retry_box.bottom - 0.003),
        (judge_box.center_x + 0.07, judge_box.top + 0.003),
        connectionstyle="arc3,rad=0.12",
    )
    _vertical_arrow(axis, judge_box, student_box)
    _vertical_arrow(axis, student_box, rl_box, linestyle="--")
    _vertical_arrow(axis, rl_box, monitor_box, linestyle="--")


def _draw_gleipnir_pipeline(axis: Axes) -> None:
    input_box = Box(0.745, 0.80, 0.32, 0.065)
    teacher_box = Box(0.745, 0.60, 0.35, 0.125)
    student_box = Box(0.745, 0.335, 0.35, 0.105)
    monitor_box = Box(0.745, 0.065, 0.30, 0.055)

    _rounded_box(
        axis,
        input_box,
        "Action-only trajectory",
        facecolor=NEUTRAL_FILL,
        edgecolor=NEUTRAL_EDGE,
    )
    _rounded_box(
        axis,
        teacher_box,
        GLEIPNIR_STAGES[0],
        facecolor=GLEIPNIR_FILL,
        edgecolor=GLEIPNIR_COLOR,
    )
    _rounded_box(
        axis,
        student_box,
        GLEIPNIR_STAGES[1],
        facecolor=GLEIPNIR_FILL,
        edgecolor=GLEIPNIR_COLOR,
    )
    _rounded_box(
        axis,
        monitor_box,
        "Deployable action-only monitor",
        facecolor=NEUTRAL_FILL,
        edgecolor=INK,
        fontsize=11.1,
    )

    _vertical_arrow(axis, input_box, teacher_box)
    _vertical_arrow(axis, teacher_box, student_box, color=GLEIPNIR_COLOR)
    _vertical_arrow(axis, student_box, monitor_box)


def plot_pipeline_comparison() -> Figure:
    """Build the publication-ready distillation-pipeline comparison."""
    set_plot_style()
    figure, axis = plt.subplots(figsize=(16.0, 9.5))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    _panel_background(axis, x=0.025, width=0.46, color=PAPER_COLOR)
    _panel_background(axis, x=0.515, width=0.46, color=GLEIPNIR_COLOR)
    axis.plot(
        [0.5, 0.5],
        [0.055, 0.91],
        transform=axis.transAxes,
        color="#D6DCE4",
        linewidth=1.2,
        zorder=1,
    )

    _panel_heading(
        axis,
        center_x=0.255,
        label="(a)",
        title="Filtered rationale distillation",
        subtitle="Sinha et al.",
        color=PAPER_COLOR,
    )
    _panel_heading(
        axis,
        center_x=0.745,
        label="(b)",
        title="Direct decision-distribution distillation",
        subtitle="Gleipnir (ours)",
        color=GLEIPNIR_COLOR,
    )
    _draw_paper_pipeline(axis)
    _draw_gleipnir_pipeline(axis)

    figure.suptitle(
        "Two routes to a deployable action-only monitor",
        x=0.5,
        y=0.99,
        fontsize=21.5,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.5,
        0.952,
        "The monitored agent's private reasoning is excluded in both approaches",
        ha="center",
        va="center",
        fontsize=11.8,
        fontweight="bold",
        color=MUTED_INK,
    )
    figure.text(
        0.5,
        0.012,
        "Direct probability targets remove rationale sampling, "
        "label-conditioned retries, and external judging.",
        ha="center",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )
    figure.subplots_adjust(left=0.015, right=0.985, top=0.925, bottom=0.045)
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        action="append",
        help=(
            "Output image path; repeat for multiple formats. Defaults to both "
            "PNG and SVG."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = args.output or [DEFAULT_OUTPUT, DEFAULT_VECTOR_OUTPUT]
    for output_path in outputs:
        output = save_figure(plot_pipeline_comparison(), output_path)
        print(f"Wrote {output}.")


if __name__ == "__main__":
    main()
