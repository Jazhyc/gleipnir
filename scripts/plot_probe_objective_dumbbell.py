"""Plot hard-label versus Kimi soft-target OOD scores for probes and LoRA."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import blended_transform_factory, offset_copy

from gleipnir.analysis.plotting import save_figure, set_plot_style

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPOSITORY_ROOT / "results/probe_generalization/full01"
DEFAULT_OUTPUTS = [
    REPOSITORY_ROOT / "figures/probe_objective_ood_dumbbell.png",
    REPOSITORY_ROOT / "figures/probe_objective_ood_dumbbell.svg",
]
BACKBONES = {
    "prompted": "Probe on frozen 4B,\nmonitor prompt (5 seeds)",
    "ntp": "Probe on frozen 4B,\nraw transcript (5 seeds)",
}
# Mean pooling and last-token probes sit near the floor on OOD (NTP mean_linear
# 0.20, last token 0.21) and would stretch the axis; the README reports them.
FAMILIES = {
    "attention": "Attention",
    "rolling_w256": "Rolling attention (window 256)",
    "rolling_w64": "Rolling attention (window 64)",
    "rolling_w10": "Rolling attention (window 10)",
    "multimax": "MultiMax",
    "alphaevolve": "AlphaEvolve",
}
# Matched 4B LoRA seeds 0-2, OOD macro pAUROC@20, from the result table in
# experiments/hard_label_seeds/README.md (artifacts span several run dirs).
LORA_SEEDS = {"hard": [0.7458, 0.7131, 0.6902], "soft": [0.7869, 0.8125, 0.7768]}
GLEIPNIR_4B_OOD = 0.804
ZERO_SHOT_OOD = 0.6175

# Green/red separated in lightness so the pair stays distinct under CVD.
GAIN_COLOR = "#176D35"
LOSS_COLOR = "#EE6A50"
GAIN_TINT = "#DCEFE2"
LOSS_TINT = "#FCE1DA"
HARD_EDGE = "#4A4A4A"
SEED_COLOR = "#9A9A9A"
INK_MUTED = "#5A5A5A"
HEADER_HEIGHT = 1.6
SECTION_GAP = 0.45
# Right-hand columns, in axes fractions.
DELTA_COLUMN = (1.03, 1.19)
WINS_CENTER = 1.27
CARD_COLOR = "#F2F3F5"
STAR_SIZE = 420
LEGEND_GAP = 24.0  # pixels between the two legend boxes
CARD_PAD = 10.0  # pixels beyond the labels and the wins column
FOOTNOTE = (
    "Wins: seeds where the soft run beats the hard run started from the same "
    "initialization. Each objective uses its own validation-best training "
    "length (and, for AlphaEvolve, penalty strength). Stars mark the probe each "
    "target would pick across all families and epochs by validation-set "
    "pAUROC@20."
)
TITLE = (
    "Kimi soft targets help probes most when they read the raw transcript, "
    "and help the fine-tuned monitor about as much"
)


@dataclass(frozen=True)
class Row:
    label: str
    hard: float | None = None
    soft: float | None = None
    hard_seeds: tuple[float, ...] = ()
    soft_seeds: tuple[float, ...] = ()
    header: bool = False
    emphasis: bool = False
    # Soft-over-hard wins among same-seed pairs.
    wins: str = ""
    # Whether this row holds the objective's validation-set selection.
    hard_selected: bool = False
    soft_selected: bool = False

    @property
    def delta(self) -> float:
        assert self.hard is not None and self.soft is not None
        return self.soft - self.hard


def _seed_scores(cell: dict) -> tuple[float, ...]:
    return tuple(
        probe["macro"]["pauroc_at_20"] for probe in cell["ood_per_probe"].values()
    )


def _paired_wins(hard: tuple[float, ...], soft: tuple[float, ...]) -> str:
    wins = sum(s > h for h, s in zip(hard, soft, strict=True))
    return f"{wins}/{len(hard)}"


def _is_selection(cell: dict, selected: dict) -> bool:
    return (cell["architecture"], cell["epoch"]) == (
        selected["architecture"],
        selected["epoch"],
    )


def _row(label: str, hard: dict, soft: dict, hard_pick: dict, soft_pick: dict) -> Row:
    hard_seeds, soft_seeds = _seed_scores(hard), _seed_scores(soft)
    return Row(
        label=label,
        hard=hard["ood_seed_median_pauroc_at_20"],
        soft=soft["ood_seed_median_pauroc_at_20"],
        hard_seeds=hard_seeds,
        soft_seeds=soft_seeds,
        # Seeds pair even where each objective picked its own epoch or
        # AlphaEvolve penalty: same-seed probes share initialization.
        wins=_paired_wins(hard_seeds, soft_seeds),
        hard_selected=_is_selection(hard, hard_pick),
        soft_selected=_is_selection(soft, soft_pick),
    )


def load_rows(results_dir: Path) -> list[Row]:
    rows: list[Row] = []
    for backbone, title in BACKBONES.items():
        pooled = json.loads((results_dir / f"ood_{backbone}.json").read_text())[
            "pooled"
        ]
        hard, soft = pooled["hard"], pooled["soft"]
        rows.append(Row(title, header=True))
        family_rows = [
            _row(
                label,
                hard["families"][family],
                soft["families"][family],
                hard["selected"],
                soft["selected"],
            )
            for family, label in FAMILIES.items()
        ]
        # Each validation-set selection must be one of the plotted families.
        assert sum(row.hard_selected for row in family_rows) == 1
        assert sum(row.soft_selected for row in family_rows) == 1
        rows.extend(family_rows)
    rows.append(Row("Fine-tuned 4B monitor\n(LoRA, 3 seeds)", header=True))
    rows.append(
        Row(
            "Gleipnir 4B recipe",
            hard=median(LORA_SEEDS["hard"]),
            soft=median(LORA_SEEDS["soft"]),
            hard_seeds=tuple(LORA_SEEDS["hard"]),
            soft_seeds=tuple(LORA_SEEDS["soft"]),
            emphasis=True,
            wins=_paired_wins(tuple(LORA_SEEDS["hard"]), tuple(LORA_SEEDS["soft"])),
        )
    )
    return rows


def plot_dumbbell(rows: list[Row]) -> plt.Figure:
    set_plot_style()
    figure, axis = plt.subplots(figsize=(11.5, 9.6))
    # Two-line section headers get a taller band than data rows.
    # Sections are separated by a gap so each can sit on its own card.
    positions = []
    cursor = 0.0
    for index, row in enumerate(rows):
        height = HEADER_HEIGHT if row.header else 1.0
        if row.header and index:
            cursor -= SECTION_GAP
        positions.append(cursor - height / 2)
        cursor -= height
    top = -cursor
    positions = [y + top for y in positions]
    delta_transform = blended_transform_factory(axis.transAxes, axis.transData)

    for y, row in zip(positions, rows, strict=True):
        if row.header:
            continue
        color = GAIN_COLOR if row.delta >= 0 else LOSS_COLOR
        axis.scatter(
            row.hard_seeds,
            [y] * len(row.hard_seeds),
            s=14,
            facecolor="none",
            edgecolor=SEED_COLOR,
            linewidth=0.9,
            zorder=2,
        )
        axis.scatter(
            row.soft_seeds,
            [y] * len(row.soft_seeds),
            s=14,
            color=SEED_COLOR,
            alpha=0.55,
            linewidth=0,
            zorder=2,
        )
        axis.plot(
            [row.hard, row.soft],
            [y, y],
            color=color,
            linewidth=3,
            solid_capstyle="round",
            zorder=3,
        )
        axis.scatter(
            [row.hard],
            [y],
            s=STAR_SIZE if row.hard_selected else 120,
            marker="*" if row.hard_selected else "o",
            facecolor="white",
            edgecolor=HARD_EDGE,
            linewidth=2,
            zorder=4,
        )
        axis.scatter(
            [row.soft],
            [y],
            s=STAR_SIZE if row.soft_selected else 130,
            marker="*" if row.soft_selected else "o",
            color=color,
            edgecolor="white",
            linewidth=1.5,
            zorder=5,
        )
        # Fixed-width badges with right-aligned values line up the decimals.
        axis.add_patch(
            FancyBboxPatch(
                (DELTA_COLUMN[0], y - 0.32),
                DELTA_COLUMN[1] - DELTA_COLUMN[0],
                0.64,
                boxstyle="round,pad=0,rounding_size=0.006",
                transform=delta_transform,
                facecolor=GAIN_TINT if row.delta >= 0 else LOSS_TINT,
                edgecolor="none",
                clip_on=False,
            )
        )
        weight = "bold" if row.emphasis else "normal"
        axis.text(
            DELTA_COLUMN[1] - 0.012,
            y,
            f"{row.delta:+.3f}".replace("-", "\u2212"),
            transform=delta_transform,
            va="center",
            ha="right",
            fontsize=13,
            color="#222222",
            fontweight=weight,
        )
        axis.text(
            WINS_CENTER,
            y,
            row.wins or "\u2013",
            transform=delta_transform,
            va="center",
            ha="center",
            fontsize=13,
            color="#222222" if row.wins else INK_MUTED,
            fontweight=weight,
        )

    for x, header, align in (
        (DELTA_COLUMN[1] - 0.012, "\u0394 soft \u2212 hard", "right"),
        (WINS_CENTER, "Wins", "center"),
    ):
        axis.text(
            x,
            positions[0],
            header,
            transform=delta_transform,
            va="center",
            ha=align,
            fontsize=12,
            color=INK_MUTED,
        )

    axis.set_yticks(positions)
    axis.set_yticklabels([row.label for row in rows])
    for tick, row in zip(axis.get_yticklabels(), rows, strict=True):
        if row.header:
            tick.set_fontweight("bold")
            tick.set_fontsize(14)
        else:
            tick.set_fontweight("bold" if row.emphasis else "normal")
            tick.set_fontsize(13)
            tick.set_color("#222222")
    axis.tick_params(axis="y", length=0)
    axis.tick_params(axis="x", labelsize=13)
    for label in axis.get_xticklabels():
        label.set_fontweight("normal")
    axis.grid(axis="y", visible=False)
    axis.grid(axis="x", color="#DCDCDC", linewidth=1)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color("#BBBBBB")

    axis.set_xlim(0.575, 0.84)
    axis.set_ylim(-0.2, top + 0.2)
    for x, label, style in (
        (ZERO_SHOT_OOD, "Base 4B zero-shot", ":"),
        (GLEIPNIR_4B_OOD, "Gleipnir 4B v0.4", "--"),
    ):
        axis.axvline(x, color=INK_MUTED, linestyle=style, linewidth=1.5, zorder=1)
        axis.text(
            x,
            top + 0.2,
            f"{label} {x:.3f}",
            rotation=0,
            ha="center",
            va="bottom",
            fontsize=11.5,
            color=INK_MUTED,
        )
    axis.set_xlabel("OOD pAUROC@20, seed median (higher is better)", fontsize=14)
    target_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=10,
            markerfacecolor="white",
            markeredgecolor=HARD_EDGE,
            markeredgewidth=2,
            label="Hard labels",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=10,
            # Neutral fill: the plotted soft marker takes the change color.
            markerfacecolor=HARD_EDGE,
            markeredgecolor=HARD_EDGE,
            label="Kimi soft targets",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=5,
            markerfacecolor=SEED_COLOR,
            markeredgecolor=SEED_COLOR,
            label="Individual seeds",
        ),
        Line2D(
            [],
            [],
            marker="*",
            linestyle="",
            markersize=15,
            markerfacecolor="white",
            markeredgecolor=HARD_EDGE,
            markeredgewidth=1.5,
            label="Chosen on validation set",
        ),
    ]
    change_handles = [
        Line2D([], [], color=GAIN_COLOR, linewidth=3, label="Soft better"),
        Line2D([], [], color=LOSS_COLOR, linewidth=3, label="Hard better"),
    ]
    legend_style = dict(
        frameon=True,
        fontsize=12,
        title_fontsize=12.5,
        borderpad=0.6,
        handletextpad=0.5,
        columnspacing=1.2,
        borderaxespad=0,
    )
    figure.tight_layout()
    left, right = _content_span(figure, axis)
    _add_section_cards(figure, axis, rows, positions, left, right)
    _add_title(figure, axis, TITLE, left, right)
    # The two legends sit side by side, centered under the cards; the footnote
    # starts at the cards' left edge.
    to_figure = figure.transFigure.inverted()
    card_left = to_figure.transform((left - CARD_PAD, 0))[0]
    renderer = figure.canvas.get_renderer()
    legends = []
    for handles, title, ncol in (
        (target_handles, "Training target", 2),
        (change_handles, "Change", 1),
    ):
        legend = figure.legend(
            handles=handles,
            title=title,
            loc="upper left",
            bbox_to_anchor=(0.0, 0.0),
            ncol=ncol,
            **legend_style,
        )
        legend.get_title().set_fontweight("bold")
        legends.append(legend)
    widths = [legend.get_window_extent(renderer).width for legend in legends]
    x = (left + right) / 2 - (sum(widths) + LEGEND_GAP) / 2
    for legend, width in zip(legends, widths, strict=True):
        # Anchor through transFigure, which tight-bbox saving updates in place.
        legend.set_bbox_to_anchor(
            (to_figure.transform((x, 0))[0], 0.0), transform=figure.transFigure
        )
        x += width + LEGEND_GAP
    # Start the footnote a little below the legends.
    legend_bottom = min(
        legend.get_window_extent(renderer).y0 for legend in figure.legends
    )
    footnote = figure.text(
        card_left,
        figure.transFigure.inverted().transform((0, legend_bottom - 22))[1],
        "",
        ha="left",
        va="top",
        fontsize=11,
        color=INK_MUTED,
        fontweight="normal",
    )
    _wrap_to_width(footnote, FOOTNOTE, right - left + 2 * CARD_PAD)
    return figure


def _content_span(figure: plt.Figure, axis: plt.Axes) -> tuple[float, float]:
    """Return the pixel span from the row labels to the Δ badge column."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    left = min(label.get_window_extent(renderer).x0 for label in axis.get_yticklabels())
    right = max(
        (text.get_bbox_patch() or text).get_window_extent(renderer).x1
        for text in axis.texts
    )
    return left, right


def _add_section_cards(
    figure: plt.Figure,
    axis: plt.Axes,
    rows: list[Row],
    positions: list[float],
    left: float,
    right: float,
) -> None:
    """Shade each section from its header to its last row, labels included."""
    x0, x1 = figure.transFigure.inverted().transform(
        [(left - CARD_PAD, 0), (right + CARD_PAD, 0)]
    )[:, 0]
    transform = blended_transform_factory(figure.transFigure, axis.transData)
    starts = [index for index, row in enumerate(rows) if row.header]
    for start, end in zip(starts, starts[1:] + [len(rows)], strict=True):
        card_top = positions[start] + HEADER_HEIGHT / 2
        card_bottom = positions[end - 1] - 0.5
        axis.add_patch(
            FancyBboxPatch(
                (x0, card_bottom),
                x1 - x0,
                card_top - card_bottom,
                boxstyle="round,pad=0,rounding_size=0.004",
                transform=transform,
                facecolor=CARD_COLOR,
                edgecolor="none",
                clip_on=False,
                zorder=0,
            )
        )


def _add_title(
    figure: plt.Figure, axis: plt.Axes, title: str, left: float, right: float
) -> None:
    """Place the title across the row labels and the Δ badge column."""
    to_axes = axis.transAxes.inverted()
    artist = axis.text(
        to_axes.transform((left, 0))[0],
        1.0,
        "",
        transform=offset_copy(axis.transAxes, figure, y=34, units="points"),
        ha="left",
        va="bottom",
        fontsize=17,
        fontweight="bold",
    )
    _wrap_to_width(artist, title, right - left)


def _wrap_to_width(artist: plt.Text, text: str, width: float) -> None:
    """Greedily word-wrap ``text`` so each rendered line fits ``width`` pixels."""
    renderer = artist.figure.canvas.get_renderer()
    lines: list[str] = []
    for word in text.split():
        candidate = f"{lines[-1]} {word}" if lines else word
        artist.set_text(candidate)
        # Glyph hinting makes text slightly wider at the 300-dpi save than at
        # the screen dpi measured here, so keep a small margin.
        if lines and artist.get_window_extent(renderer).width > 0.96 * width:
            lines.append(word)
        elif lines:
            lines[-1] = candidate
        else:
            lines.append(word)
    artist.set_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Probe study run directory with ood_<backbone>.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        action="append",
        help="Output image path; repeat for multiple formats. "
        "Defaults to both PNG and SVG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results_dir)
    for output_path in args.output or DEFAULT_OUTPUTS:
        # Extra padding keeps a white margin beside the full-width cards.
        output = save_figure(plot_dumbbell(rows), output_path, pad_inches=0.4)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
