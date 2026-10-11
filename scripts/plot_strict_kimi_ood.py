"""Plot cached strict-filtered student versus Kimi K3 on matched OOD rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from gleipnir.data.monitoring import file_hash, read_rows, write_json
from gleipnir.plotting import plt, save_figure, set_plot_style

ROOT = Path(__file__).resolve().parents[1]
INPUTS = dict(
    [
        (
            "results/teacher-disagreement-strict-ood01/evaluation/ood.jsonl",
            "daee33441f55c8460d3c50c6310a9cde21ba8dc03114b07e81aeae413caf9e68",
        ),
        (
            "results/tool_trajectory_monitoring/kimi_k3_ood_canary_fireworks.jsonl",
            "c268958b594e4482f539ccaeb371454f629faeff1f9af96064a69d8bc99003bb",
        ),
        (
            "results/tool_trajectory_monitoring/kimi_k3_ood_full_remaining_fireworks.jsonl",
            "69e826d425830b2053b7d0055aae03c907b29e9e5f621f8c5b44a941c44b5153",
        ),
    ]
)
STEM = ROOT / "figures/strict_filtered_kimi_k3_ood"


def pair_scores(student: list[dict], teacher: list[dict]) -> list[dict]:
    """Require one-to-one coverage, matching labels, source and trajectory bytes."""
    by_id = {row["id"]: row for row in teacher}
    if (
        not student
        or len(by_id) != len(teacher)
        or len({r["id"] for r in student}) != len(student)
        or {r["id"] for r in student} != set(by_id)
    ):
        raise ValueError("paired ID coverage/uniqueness drift")
    paired = []
    for row in student:
        other = by_id[row["id"]]
        for field in ("ground_truth", "source_dataset", "trajectory_sha256"):
            if row[field] != other["metadata"][field]:
                raise ValueError(f"paired {field} drift")
        if row["ground_truth"] not in (0, 1):
            raise ValueError("invalid binary source label")
        scores = np.array([row["score"], other["score"]], dtype=float)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("invalid probability")
        paired.append(
            {
                "id": row["id"],
                "source": row["source_dataset"],
                "label": row["ground_truth"],
                "student": float(scores[0]),
                "kimi": float(scores[1]),
            }
        )
    return paired


def describe(rows: list[dict]) -> dict:
    student = np.array([r["student"] for r in rows])
    kimi = np.array([r["kimi"] for r in rows])
    counts = np.histogram2d(kimi >= 0.5, student >= 0.5, bins=[[-0.5, 0.5, 1.5]] * 2)[0]
    hist_s = np.histogram(student, bins=np.linspace(0, 1, 21))[0] / len(rows)
    hist_k = np.histogram(kimi, bins=np.linspace(0, 1, 21))[0] / len(rows)
    return {
        "n": len(rows),
        "pearson": float(np.corrcoef(student, kimi)[0, 1]),
        "spearman": float(spearmanr(student, kimi).statistic),
        "mean_absolute_score_difference": float(np.abs(student - kimi).mean()),
        "mean_student_minus_kimi": float((student - kimi).mean()),
        "agreement_at_05": float(np.mean((student >= 0.5) == (kimi >= 0.5))),
        "within_005": float(np.mean(np.abs(student - kimi) <= 0.05)),
        "within_010": float(np.mean(np.abs(student - kimi) <= 0.1)),
        "histogram_overlap_bin_width_005": float(np.minimum(hist_s, hist_k).sum()),
        "decision_counts_kimi_rows_student_columns": counts.astype(int).tolist(),
        "student_mean": float(student.mean()),
        "kimi_mean": float(kimi.mean()),
    }


def main() -> None:
    for path, expected in INPUTS.items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"frozen prediction checksum changed: {path}")
    paths = list(INPUTS)
    rows = pair_scores(
        read_rows(ROOT / paths[0]),
        read_rows(ROOT / paths[1]) + read_rows(ROOT / paths[2]),
    )
    if len(rows) != 6395:
        raise ValueError("expected all 6395 OOD examples")
    overall = describe(rows)
    summary = {
        "pooled": overall,
        "by_label": {
            str(label): describe([r for r in rows if r["label"] == label])
            for label in (0, 1)
        },
        "by_source": {
            source: describe([r for r in rows if r["source"] == source])
            for source in sorted({r["source"] for r in rows})
        },
    }
    set_plot_style()
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    colors = ["#2878B5", "#D55E00"]
    for ax, label in zip(axes[0], (0, 1), strict=True):
        group = [r for r in rows if r["label"] == label]
        for field, name, color in zip(
            ("student", "kimi"), ("Strict-filtered 4B", "Kimi K3"), colors, strict=True
        ):
            ax.hist(
                [r[field] for r in group],
                bins=np.linspace(0, 1, 21),
                weights=np.full(len(group), 100 / len(group)),
                histtype="step",
                linewidth=2.5,
                color=color,
                label=name,
            )
        ax.set(
            title=f"{'Benign' if label == 0 else 'Harmful'} trajectories "
            f"(n={len(group):,})",
            xlabel="P(harmful)",
            ylabel="Share within label (%)",
            xlim=(0, 1),
        )
        ax.axvline(0.5, color="#888888", linestyle="--", linewidth=1)
        ax.legend(fontsize=12)
    student = np.array([r["student"] for r in rows])
    kimi = np.array([r["kimi"] for r in rows])
    ax = axes[1, 0]
    artist = ax.hexbin(
        kimi,
        student,
        gridsize=35,
        extent=(0, 1, 0, 1),
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0.1,
    )
    ax.plot([0, 1], [0, 1], color="#D55E00", linestyle="--", linewidth=1.5)
    ax.axvline(0.5, color="#888888", linestyle=":", linewidth=1)
    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1)
    ax.set(
        title="Paired scores on the same trajectories",
        xlabel="Kimi K3 P(harmful)",
        ylabel="Strict-filtered 4B P(harmful)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    fig.colorbar(artist, ax=ax, label="Trajectories per hexagon (log scale)", pad=0.02)
    ax = axes[1, 1]
    for label, color in zip((0, 1), colors, strict=True):
        group = [r for r in rows if r["label"] == label]
        ax.hist(
            [r["student"] - r["kimi"] for r in group],
            bins=np.linspace(-1, 1, 41),
            weights=np.full(len(group), 100 / len(group)),
            histtype="step",
            linewidth=2.5,
            color=color,
            label="Benign" if label == 0 else "Harmful",
        )
    ax.axvline(0, color="#888888", linestyle="--", linewidth=1)
    ax.set(
        title="Paired score differences",
        xlabel="Strict-filtered 4B score − Kimi K3 score",
        ylabel="Share within label (%)",
        xlim=(-1, 1),
    )
    ax.legend(fontsize=12)
    fig.suptitle(
        "Strict-filtered Gleipnir 4B versus Kimi K3 · OOD", fontsize=21, y=1.015
    )
    fig.text(
        0.5,
        -0.035,
        "All 6,395 matched trajectories · decision agreement at 0.5: "
        f"{overall['agreement_at_05']:.1%} · Spearman ρ: {overall['spearman']:.3f}"
        " · mean absolute score difference: "
        f"{overall['mean_absolute_score_difference']:.3f}\n"
        "Student: compact prompt, merged BF16 · Kimi: detailed teacher prompt, "
        "cached Fireworks scores · histogram bin width 0.05",
        ha="center",
        fontsize=12,
        color="#555555",
    )
    fig.tight_layout()
    outputs = {}
    for suffix in ("png", "svg", "pdf"):
        path = STEM.with_suffix("." + suffix)
        save_figure(fig, path)
        outputs[str(path.relative_to(ROOT))] = file_hash(path)
    write_json(
        STEM.with_suffix(".provenance.json"),
        {
            "inputs_sha256": INPUTS,
            "script_sha256": file_hash(Path(__file__)),
            "summary": summary,
            "outputs_sha256": outputs,
            "qualification": "Pooled descriptive overlap; historical teacher "
            "interface differs. No OOD tuning or new inference.",
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
