"""Plot matched full-boundary teacher probabilities without new inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gleipnir.plotting import plt, save_figure, set_plot_style


def load_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Validate paired identities and finite binary probabilities."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Expected nonempty, unique paired IDs")
    scores = np.array([[row["qwen_score"], row["kimi_score"]] for row in rows])
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite probabilities in [0, 1]")
    return scores[:, 0], scores[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("results/teacher_agreement/paired_scores.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    qwen, kimi = load_scores(args.source)
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, scores, title, color in zip(
        axes,
        (qwen, kimi),
        ("Qwen 3.5 27B · FP8", "Kimi K3"),
        ("#4C78A8", "#E45756"),
        strict=True,
    ):
        ax.hist(
            scores,
            bins=bins,
            weights=np.full(len(scores), 100 / len(scores)),
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set(title=title, xlim=(0, 1), xlabel="Normalized P(problematic behavior)")
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.axvline(0.5, color="#555555", linewidth=1, linestyle="--", alpha=0.6)
        middle = np.mean((scores >= 0.1) & (scores <= 0.9))
        ax.text(
            0.5,
            0.94,
            f"Mean score: {scores.mean():.3f}\nScores in [0.1, 0.9]: {middle:.1%}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
        )
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of trajectories (%)")
    fig.suptitle(
        "Teacher scores at the final classification boundary", fontsize=21, y=1.02
    )
    fig.text(
        0.5,
        -0.03,
        f"Same {len(qwen):,} full trajectories · "
        "source/label-balanced training subset, not the full dataset\n"
        "Same detailed rubric · probabilities normalized over decision tokens 0 and 1 "
        "· bin width 0.05",
        ha="center",
        fontsize=12,
        color="#555555",
    )
    fig.tight_layout()
    for suffix in ("png", "svg"):
        save_figure(
            fig, args.output_dir / f"teacher_final_boundary_histograms.{suffix}"
        )
    print(
        json.dumps(
            {
                "n": len(qwen),
                "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "qwen_mean": float(qwen.mean()),
                "kimi_mean": float(kimi.mean()),
                "qwen_middle_fraction": float(np.mean((qwen >= 0.1) & (qwen <= 0.9))),
                "kimi_middle_fraction": float(np.mean((kimi >= 0.1) & (kimi <= 0.9))),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
