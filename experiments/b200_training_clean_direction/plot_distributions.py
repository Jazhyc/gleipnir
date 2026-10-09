"""Plot complete observed distributions without tail selection or KDE fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gleipnir.analysis.plotting import set_plot_style
from gleipnir.data.monitoring import file_hash, read_rows, write_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "results/b200_training_clean_direction/clean_direction01"
SOURCE_NAMES = {
    "crh": "CoT Red-Handed",
    "ctrl": "BashBench",
    "gloom": "Gloom",
    "o46": "BashArena / Opus 4.6",
    "s45": "BashArena / Sonnet 4.5",
    "stride": "STRIDE",
}
BLUE = "#0072B2"
ORANGE = "#D55E00"


def histogram(ax, values: np.ndarray, bins: np.ndarray, label: str, color: str) -> None:
    """Show a unit-area histogram, checking complete bin coverage."""
    counts, _ = np.histogram(values, bins=bins)
    if counts.sum() != len(values):
        raise ValueError("histogram drops observations")
    density, _ = np.histogram(values, bins=bins, density=True)
    if not np.isclose(np.sum(density * np.diff(bins)), 1):
        raise ValueError("histogram normalization failed")
    ax.stairs(density, bins, label=f"{label} (n={len(values):,})", color=color, lw=2)
    ax.stairs(density, bins, color=color, fill=True, alpha=0.08)


def save(fig, out: Path, stem: str) -> list[Path]:
    """Export shareable raster and vector copies of the same figure."""
    paths = [out / (stem + suffix) for suffix in [".png", ".pdf"]]
    for path in paths:
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def plot(out: Path) -> None:
    rows = read_rows(out / "paired_models.jsonl")
    if len(rows) != 8688 or any(r["view"] != "original" for r in rows):
        raise ValueError("requires the complete original-data census")
    base = np.array([r["base_z20"] for r in rows])
    trained = np.array([r["trained_z20"] for r in rows])
    delta = np.array([r["delta_z20"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    targets = np.array([r["soft_target"] for r in rows])
    sources = np.array([r["source"] for r in rows])
    if not np.isfinite([base, trained, delta, targets]).all():
        raise ValueError("nonfinite observations")
    if not np.allclose(trained - base, delta, atol=1e-12):
        raise ValueError("paired difference changed")
    projection_bins = np.linspace(
        min(base.min(), trained.min()), max(base.max(), trained.max()), 51
    )
    delta_bins = np.linspace(delta.min(), delta.max(), 51)
    set_plot_style()
    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), layout="constrained")
    ax = axes[0, 0]
    histogram(ax, base, projection_bins, "Base model", BLUE)
    histogram(ax, trained, projection_bins, "Trained SDPA model", ORANGE)
    ax.set(
        title="Activation alignment before and after training",
        xlabel="Layer-20 projection onto fixed direction",
        ylabel="Density",
    )
    ax.legend()
    ax = axes[0, 1]
    for label, color, name in [
        (0, BLUE, "Harmless-labeled"),
        (1, ORANGE, "Harmful-labeled"),
    ]:
        histogram(ax, delta[labels == label], delta_bins, name, color)
    ax.set(
        title="Training-associated change by behavior label",
        xlabel="Projection change (trained − base)",
        ylabel="Density within group",
    )
    ax.legend()
    ax = axes[1, 0]
    colors = plt.get_cmap("tab10")
    for i, (source, name) in enumerate(SOURCE_NAMES.items()):
        values = np.sort(delta[sources == source])
        x = np.r_[delta.min(), values, delta.max()]
        y = np.r_[0, np.arange(1, len(values) + 1) / len(values), 1]
        ax.step(
            x,
            y,
            where="post",
            color=colors(i),
            lw=1.8,
            label=f"{name} (n={len(values):,})",
        )
    ax.set(
        title="Full change distribution by source",
        xlabel="Projection change (trained − base)",
        ylabel="Fraction of source at or below x",
        ylim=(0, 1),
    )
    ax.legend(loc="upper left", fontsize=9)
    ax = axes[1, 1]
    for positive, color in [(False, BLUE), (True, ORANGE)]:
        mask = (labels == 0) & ((targets >= 0.5) == positive)
        histogram(
            ax,
            delta[mask],
            delta_bins,
            "Teacher target ≥0.5" if positive else "Teacher target <0.5",
            color,
        )
    ax.set(
        title="Harmless-labeled records: teacher-target groups",
        xlabel="Projection change (trained − base)",
        ylabel="Density within group",
    )
    ax.legend()
    for ax in axes.flat:
        ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.7)
    fig.suptitle(
        "All 8,688 original training records · no added synthetic injections",
        fontsize=17,
    )
    fig.supxlabel(
        "Each density curve has area 1. Includes original duplicate records. "
        "Alignment is not causal training influence.",
        fontsize=10,
    )
    artifacts = save(fig, out, "direction_distributions")

    fig, axes = plt.subplots(
        2, 3, figsize=(16, 9), sharex=True, sharey=True, layout="constrained"
    )
    for ax, (source, name) in zip(axes.flat, SOURCE_NAMES.items(), strict=True):
        for label, color, label_name in [(0, BLUE, "Harmless"), (1, ORANGE, "Harmful")]:
            histogram(
                ax,
                delta[(sources == source) & (labels == label)],
                delta_bins,
                label_name,
                color,
            )
        ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.7)
        ax.set_title(name)
        ax.legend()
    fig.suptitle(
        "Complete source distributions, separated by recorded behavior label",
        fontsize=17,
    )
    fig.supxlabel(
        "Layer-20 projection change (trained − base); identical bins and axis limits",
        fontsize=12,
    )
    fig.supylabel("Density within each source / label group", fontsize=12)
    artifacts += save(fig, out, "direction_source_distributions")
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5), sharex=True, sharey=True, layout="constrained"
    )
    for ax, label, name in zip(axes, [0, 1], ["Harmless", "Harmful"], strict=True):
        for positive, color in [(False, BLUE), (True, ORANGE)]:
            mask = (labels == label) & ((targets >= 0.5) == positive)
            histogram(
                ax,
                delta[mask],
                delta_bins,
                "Kimi K3 target ≥0.5" if positive else "Kimi K3 target <0.5",
                color,
            )
            ax.axvline(delta[mask].mean(), color=color, ls=":", lw=1.5)
        ax.axvline(0, color="gray", ls="--", lw=1)
        ax.set_title(f"{name}-labeled records")
        ax.set_xlabel("Projection change (trained − base)")
        ax.legend()
    axes[0].set_ylabel("Density within each teacher-target group")
    fig.suptitle(
        "Teacher-target split within each recorded behavior label", fontsize=16
    )
    fig.supxlabel(
        "Identical bins and scales; each curve has area 1. "
        "Colored dotted lines mark group means; gray dashed line marks zero.",
        fontsize=10,
    )
    artifacts += save(fig, out, "direction_teacher_by_label_distributions")
    write_json(
        out / "distribution_plot_receipt.json",
        {
            "input_sha256": file_hash(out / "paired_models.jsonl"),
            "script_sha256": file_hash(Path(__file__)),
            "style_sha256": file_hash(ROOT / "src/gleipnir/analysis/plotting.py"),
            "original_records": len(rows),
            "histogram_bins": 50,
            "full_range_included": True,
            "within_group_density_area": 1,
            "source_counts": {s: int((sources == s).sum()) for s in SOURCE_NAMES},
            "artifacts": {p.name: file_hash(p) for p in artifacts},
            "qualification": (
                "Post-capture descriptive visualization; no data filtering, refit, "
                "tail selection or new model requests."
            ),
        },
    )
    print(json.dumps({"records": len(rows), "artifacts": [str(p) for p in artifacts]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    plot(parser.parse_args().output)


if __name__ == "__main__":
    main()
