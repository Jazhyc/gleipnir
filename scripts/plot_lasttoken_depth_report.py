"""Report figure (Appendix F): last-token probes by depth, validation and OOD.

Prompted last-token logistic-regression probes with Kimi K3 soft targets, at
the `Prediction:` position. Lines give the validation (ID) pAUROC@20 at every
layer, the best over the regularization grid, for Qwen3.5-4B and
Qwen3.8-27B. Markers give Mean-OOD pAUROC@20 where it was measured: for the
27B at the fixed layers 24-44 of the depth follow-up (C chosen on ID per layer)
and at the layer selected on the validation set; for the 4B at its selected
layer only. The shaded band marks the half-depth layers of the sequence
probes. Relative depth is (layer + 1) / layers. Combines the talk figures
`lasttoken_depth_id` and `lasttoken_ood_depth` (branch slide-figures) without
their annotations.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gleipnir-matplotlib")
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from gleipnir.plotting import save_figure, set_plot_style

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/probe_generalization"
ASSETS = ROOT / "outputs/reports/gleipnir_v1_technical_report/assets"
MODELS = {
    "Qwen3.5-4B": {"run": "full01", "layers": 32, "color": "#4C78A8"},
    "Qwen3.8-27B": {"run": "scale27b_prompted01", "layers": 64, "color": "#E45756"},
}
OBJECTIVE = "soft"
INK_MUTED = "#6B7280"
CHANCE = 0.1


def validation_curve(run: str) -> dict[int, float]:
    best: dict[int, float] = {}
    path = RESULTS / run / "lasttoken_prompted.json"
    for probe in json.loads(path.read_text())["probes"]:
        if probe["objective"] == OBJECTIVE:
            layer, score = probe["layer"], probe["id"]["pauroc_at_20"]
            best[layer] = max(best.get(layer, 0.0), score)
    return best


def ood_points(run: str) -> list[tuple[int, float]]:
    points = []
    depth = RESULTS / run / "lasttoken_depth_prompted.json"
    if depth.exists():
        points += [
            (p["layer"], p["ood"]["pauroc_at_20"])
            for p in json.loads(depth.read_text())["probes"]
            if p["objective"] == OBJECTIVE
        ]
    selected = json.loads((RESULTS / run / "ood_prompted.json").read_text())
    chosen = selected["last_token"][OBJECTIVE]
    points.append((chosen["layer"], chosen["ood"]["pauroc_at_20"]))
    return sorted(points)


def plot() -> plt.Figure:
    set_plot_style()
    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    axis.axvspan(0.46, 0.52, color="#EEF1F4", zorder=0)
    axis.text(
        0.49, 0.15, "half-depth\nprobes", color=INK_MUTED, ha="center", fontsize=12
    )
    for spec in MODELS.values():
        n, color = spec["layers"], spec["color"]
        curve = validation_curve(spec["run"])
        layers = sorted(curve)
        axis.plot(
            [(layer + 1) / n for layer in layers],
            [curve[layer] for layer in layers],
            color=color,
            linewidth=2.2,
            zorder=2,
        )
        points = ood_points(spec["run"])
        axis.plot(
            [(layer + 1) / n for layer, _ in points],
            [score for _, score in points],
            color=color,
            linewidth=1.8,
            linestyle=(0, (4, 2.5)),
            marker="o",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=3,
        )
    # pAUROC@20 normalizes by the FPR width, so a random monitor scores 0.1.
    axis.axhline(CHANCE, color=INK_MUTED, linewidth=1.4, linestyle=":", zorder=1)
    axis.text(
        0.99, CHANCE + 0.015, "chance", color=INK_MUTED, ha="right", fontsize=12
    )
    axis.set_xlim(0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axis.set_xticklabels(["input", "25%", "50%", "75%", "output"])
    axis.set_xlabel("Relative depth of the probed layer", fontsize=14)
    axis.set_ylabel("pAUROC@20", fontsize=14)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    handles = [
        Line2D([], [], color=spec["color"], linewidth=2.5, label=name)
        for name, spec in MODELS.items()
    ] + [
        Line2D([], [], color=INK_MUTED, linewidth=2.2, label="Validation set"),
        Line2D(
            [],
            [],
            color=INK_MUTED,
            linewidth=1.8,
            linestyle=(0, (4, 2.5)),
            marker="o",
            markersize=7,
            markeredgecolor="white",
            label="OOD suite (measured layers)",
        ),
    ]
    axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        prop={"size": 12.5, "weight": "bold"},
        handlelength=2.4,
        columnspacing=1.4,
    )
    figure.tight_layout()
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, action="append")
    args = parser.parse_args()
    outputs = args.output or [
        ASSETS / "lasttoken_depth.pdf",
        ASSETS / "lasttoken_depth.svg",
    ]
    for output in outputs:
        print(f"wrote {save_figure(plot(), output)}")


if __name__ == "__main__":
    main()
