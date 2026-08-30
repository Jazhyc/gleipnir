"""Shared plotting conventions and helpers for Gleipnir figures."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gleipnir-matplotlib")
)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

ORIGIN_PALETTE = {
    "Paper": "#4C78A8",
    "Gleipnir": "#E45756",
}


def set_plot_style() -> None:
    """Apply the repository's readable, colorblind-safe plotting defaults."""
    sns.set_theme(
        context="talk",
        style="whitegrid",
        palette="colorblind",
        rc={
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
        },
    )


def pareto_frontier_mask(
    costs: np.ndarray | list[float],
    performances: np.ndarray | list[float],
) -> np.ndarray:
    """Return nondominated points for lower cost and higher performance."""
    cost = np.asarray(costs, dtype=float)
    performance = np.asarray(performances, dtype=float)
    if cost.ndim != 1 or cost.shape != performance.shape or not len(cost):
        raise ValueError("costs and performances must be nonempty paired vectors")
    if not np.isfinite(cost).all() or not np.isfinite(performance).all():
        raise ValueError("costs and performances must be finite")
    if (cost <= 0).any():
        raise ValueError("costs must be positive")

    frontier = np.ones(len(cost), dtype=bool)
    for index, (point_cost, point_performance) in enumerate(
        zip(cost, performance, strict=True)
    ):
        no_worse = (cost <= point_cost) & (performance >= point_performance)
        strictly_better = (cost < point_cost) | (performance > point_performance)
        frontier[index] = not np.any(no_worse & strictly_better)
    return frontier


def save_figure(figure: Figure, output_path: str | Path) -> Path:
    """Save a tightly cropped figure, creating its parent directory."""
    output = Path(output_path)
    if not output.suffix:
        raise ValueError("output_path must include a file extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    if output.suffix.lower() == ".svg":
        svg = output.read_text(encoding="utf-8")
        output.write_text(
            "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
            encoding="utf-8",
        )
    return output
