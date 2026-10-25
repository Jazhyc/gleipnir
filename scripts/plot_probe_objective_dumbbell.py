"""Plot hard-label versus Kimi soft-target OOD scores for probes and LoRA."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from gleipnir.analysis.plotting import save_figure, set_plot_style

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPOSITORY_ROOT / "results/probe_generalization/full01"
DEFAULT_OUTPUTS = [
    REPOSITORY_ROOT / "figures/probe_objective_ood_dumbbell.png",
    REPOSITORY_ROOT / "figures/probe_objective_ood_dumbbell.svg",
]
BACKBONES = {
    "prompted": "Probe on frozen 4B, monitor prompt (5 seeds)",
    "ntp": "Probe on frozen 4B, raw transcript (5 seeds)",
}
# Mean pooling and last-token probes sit near the floor on OOD (NTP mean_linear
# 0.20, last token 0.21) and would stretch the axis; the README reports them.
FAMILIES = {
    "attention": "Attention",
    "rolling_w256": "Rolling attention (window 256)",
    "rolling_w64": "Rolling attention (window 64)",
    "rolling_w10": "Rolling attention (window 10)",
    "multimax": "MultiMax",
    "alphaevolve": "AlphaEvolve-discovered probe",
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


@dataclass(frozen=True)
class Row:
    label: str
    hard: float | None = None
    soft: float | None = None
    hard_seeds: tuple[float, ...] = ()
    soft_seeds: tuple[float, ...] = ()
    header: bool = False
    emphasis: bool = False
    # Soft-over-hard wins among same-seed pairs; empty when the cells differ
    # in architecture, so seed indices do not pair.
    wins: str = ""

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


def _row(
    label: str, hard: dict, soft: dict, *, paired: bool = True, emphasis: bool = False
) -> Row:
    hard_seeds, soft_seeds = _seed_scores(hard), _seed_scores(soft)
    # Family rows pair seeds even where each objective picked its own epoch or
    # AlphaEvolve penalty; the selected cells are different architectures.
    wins = _paired_wins(hard_seeds, soft_seeds) if paired else ""
    return Row(
        label=label,
        hard=hard["ood_seed_median_pauroc_at_20"],
        soft=soft["ood_seed_median_pauroc_at_20"],
        hard_seeds=hard_seeds,
        soft_seeds=soft_seeds,
        emphasis=emphasis,
        wins=wins,
    )


def load_rows(results_dir: Path) -> list[Row]:
    rows: list[Row] = []
    for backbone, title in BACKBONES.items():
        pooled = json.loads((results_dir / f"ood_{backbone}.json").read_text())[
            "pooled"
        ]
        hard, soft = pooled["hard"], pooled["soft"]
        rows.append(Row(title, header=True))
        rows.append(
            _row(
                "Selected using validation set",
                hard["selected"],
                soft["selected"],
                paired=False,
                emphasis=True,
            )
        )
        rows.extend(
            _row(label, hard["families"][family], soft["families"][family])
            for family, label in FAMILIES.items()
        )
    rows.append(Row("Fine-tuned 4B monitor (LoRA, 3 seeds)", header=True))
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
    positions = list(range(len(rows)))[::-1]
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
            s=120,
            facecolor="white",
            edgecolor=HARD_EDGE,
            linewidth=2,
            zorder=4,
        )
        axis.scatter(
            [row.soft],
            [y],
            s=130,
            color=color,
            edgecolor="white",
            linewidth=1.5,
            zorder=5,
        )
        axis.text(
            1.015,
            y,
            f"{row.delta:+.3f}".replace("-", "\u2212")
            + (f" \u00b7 {row.wins}" if row.wins else ""),
            transform=delta_transform,
            va="center",
            ha="left",
            fontsize=13,
            color="#222222",
            fontweight="bold" if row.emphasis else "normal",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=GAIN_TINT if row.delta >= 0 else LOSS_TINT,
                edgecolor="none",
            ),
        )

    axis.text(
        1.015,
        positions[0],
        "Δ soft − hard · wins",
        transform=delta_transform,
        va="center",
        ha="left",
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
    axis.grid(axis="x", color="#ECECEC", linewidth=1)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color("#BBBBBB")
    for y, row in zip(positions, rows, strict=True):
        if row.header and y != positions[0]:
            axis.axhline(y + 0.5, color="#D0D0D0", linewidth=1)

    axis.set_xlim(0.575, 0.84)
    axis.set_ylim(-0.7, len(rows) - 0.3)
    for x, label, style in (
        (ZERO_SHOT_OOD, "Base 4B zero-shot", ":"),
        (GLEIPNIR_4B_OOD, "Gleipnir 4B v0.4", "--"),
    ):
        axis.axvline(x, color=INK_MUTED, linestyle=style, linewidth=1.5, zorder=1)
        axis.text(
            x,
            len(rows) - 0.3,
            f"{label} {x:.3f}",
            rotation=0,
            ha="center",
            va="bottom",
            fontsize=11.5,
            color=INK_MUTED,
        )
    axis.set_xlabel("OOD pAUROC@20, seed median (higher is better)", fontsize=14)
    axis.set_title(
        "Kimi soft targets help probes most when they read the raw\n"
        "transcript, and help the fine-tuned monitor about as much",
        fontsize=17,
        fontweight="bold",
        loc="left",
        pad=34,
    )

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
            markerfacecolor=GAIN_COLOR,
            markeredgecolor="white",
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
    )
    figure.tight_layout()
    for handles, title, anchor, loc, ncol in (
        (target_handles, "Training target", (0.02, 0.0), "upper left", 3),
        (change_handles, "Change", (0.98, 0.0), "upper right", 2),
    ):
        legend = figure.legend(
            handles=handles,
            title=title,
            loc=loc,
            bbox_to_anchor=anchor,
            ncol=ncol,
            **legend_style,
        )
        legend.get_title().set_fontweight("bold")
    figure.text(
        0.02,
        -0.105,
        "Wins: seeds where the soft run beats the hard run started from the "
        "same initialization. Each objective uses its own\nvalidation-best "
        "training length (and, for AlphaEvolve, penalty strength). Selected "
        "rows compare different probe types, so seeds do not pair.",
        ha="left",
        va="top",
        fontsize=11,
        color=INK_MUTED,
        fontweight="normal",
    )
    return figure


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
        output = save_figure(plot_dumbbell(rows), output_path)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
