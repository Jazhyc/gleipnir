"""Compute and plot calibration on the frozen matched training subset."""

import hashlib
import json
from pathlib import Path

from gleipnir.calibration import binary_calibration
from gleipnir.plotting import plt, save_figure, set_plot_style


def main() -> None:
    source = Path("results/teacher_agreement/paired_scores.jsonl")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate paired IDs")
    result = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": "Source/label-balanced training diagnostic, not held-out calibration",
        "ece_definition": "Positive-class equal-width bins; weighted absolute gap",
        "log_loss_clip": 1e-15,
        "teachers": {},
    }
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, teacher, name, color in zip(
        axes,
        ("qwen", "kimi"),
        ("Qwen 3.5 27B · FP8", "Kimi K3"),
        ("#4C78A8", "#E45756"),
        strict=True,
    ):
        labels = [r["label"] for r in rows]
        probabilities = [r[f"{teacher}_score"] for r in rows]
        stats = binary_calibration(labels, probabilities)
        stats["ece_bin_sensitivity"] = {
            str(n): binary_calibration(labels, probabilities, n)["ece"]
            for n in (5, 10, 20)
        }
        stats["sources"] = {
            s: binary_calibration(
                [r["label"] for r in rows if r["source"] == s],
                [r[f"{teacher}_score"] for r in rows if r["source"] == s],
            )
            for s in sorted({r["source"] for r in rows})
        }
        result["teachers"][teacher] = stats
        ax.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
        for b in stats["bins"]:
            if not b["n"]:
                continue
            p, y = b["mean_probability"], b["observed_rate"]
            low, high = b["wilson95"]
            ax.errorbar(
                p,
                y,
                yerr=[[max(0, y - low)], [max(0, high - y)]],
                fmt="o",
                color=color,
                capsize=3,
                markersize=7,
            )
            ax.annotate(
                str(b["n"]),
                (p, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                color=color,
            )
        ax.set(
            title=f"{name}\nECE (10 bins): {stats['ece']:.3f}",
            xlim=(-0.03, 1.04),
            ylim=(-0.03, 1.04),
            xlabel="Mean predicted P(problematic)",
        )
    axes[0].set_ylabel("Observed fraction problematic")
    fig.suptitle(
        "Final-boundary reliability · 640 matched training samples", fontsize=19
    )
    fig.text(
        0.5,
        -0.03,
        "10 equal-width bins · labels show bin counts · "
        "bars: descriptive 95% Wilson intervals\n"
        "Source/label-balanced subset, not held-out; "
        "intervals assume independent samples",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout()
    for suffix in ("png", "svg"):
        save_figure(fig, f"figures/teacher_final_boundary_calibration.{suffix}")
    Path("results/teacher_agreement/calibration.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                t: {k: v for k, v in s.items() if k not in ("bins", "sources")}
                for t, s in result["teachers"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
