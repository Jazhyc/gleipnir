"""Label calibration of unadapted Qwen3.8-27B and Kimi K3 on the training rows.

Uses the row-id join of ``compare_id_train.py`` (prompt hashes checked) over
all 8,688 training rows, plus the 640-row source/label-balanced subset of the
earlier Qwen3.5-27B comparison (``experiments/teacher_agreement``) for a
like-for-like ECE. Training rows are Kimi's distillation inputs, so this is a
diagnostic, not held-out calibration.
"""

from __future__ import annotations

import json

from experiments.qwen27b_generations_ood.compare_id_train import KIMI, OUT, ROOT, rows
from gleipnir.calibration import binary_calibration
from gleipnir.data.monitoring import file_hash
from gleipnir.plotting import plt, save_figure, set_plot_style

PROMPTS = ROOT / "data/tool_trajectory_monitoring/teacher_training/prompts.jsonl"
QWEN = OUT / "qwen38_27b_train/predictions.jsonl"
MATCHED = ROOT / "results/teacher_agreement/paired_scores.jsonl"
TEACHERS = (("qwen", "Qwen3.8-27B · BF16", "#4C78A8"), ("kimi", "Kimi K3", "#E45756"))


def joined() -> list[dict]:
    qwen = {r["id"]: r for r in rows(QWEN)}
    kimi = {str(r["index"]): r for r in rows(KIMI)}
    result = []
    for row in rows(PROMPTS):
        rid, meta = row["id"], row["metadata"]
        if (
            kimi[rid]["rendered_prompt_sha256"] != meta["rendered_prompt_sha256"]
            or qwen[rid]["source_prompt_sha256"] != meta["rendered_prompt_sha256"]
        ):
            raise ValueError(f"prompt identity drift: {rid}")
        result.append(
            {
                "id": rid,
                "source": meta["source_dataset"],
                "label": int(meta["ground_truth"]),
                "qwen": float(qwen[rid]["score"]),
                "kimi": float(kimi[rid]["soft_target"]),
            }
        )
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("duplicate training row ids")
    return result


def calibration(records: list[dict], teacher: str) -> dict:
    labels = [r["label"] for r in records]
    scores = [r[teacher] for r in records]
    stats = binary_calibration(labels, scores)
    stats["ece_bin_sensitivity"] = {
        str(n): binary_calibration(labels, scores, n)["ece"] for n in (5, 10, 20)
    }
    stats["sources"] = {
        s: binary_calibration(
            [r["label"] for r in records if r["source"] == s],
            [r[teacher] for r in records if r["source"] == s],
        )
        for s in sorted({r["source"] for r in records})
    }
    return stats


def main() -> None:
    records = joined()
    matched_ids = {r["id"] for r in rows(MATCHED)}
    matched = [r for r in records if r["id"] in matched_ids]
    if len(matched) != len(matched_ids):
        raise ValueError("matched subset is not covered by the training join")
    report = {
        "scope": "Training rows (Kimi's distillation inputs); not held-out",
        "ece_definition": "Positive-class equal-width bins; weighted absolute gap",
        "inputs_sha256": {
            str(p.relative_to(ROOT)): file_hash(p)
            for p in (PROMPTS, QWEN, KIMI, MATCHED)
        },
        "all": {t: calibration(records, t) for t, _, _ in TEACHERS},
        "matched_640": {t: calibration(matched, t) for t, _, _ in TEACHERS},
    }
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, (teacher, name, color) in zip(axes, TEACHERS, strict=True):
        stats = report["all"][teacher]
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
        f"Final-boundary reliability · {len(records):,} training rows", fontsize=19
    )
    fig.text(
        0.5,
        -0.03,
        "10 equal-width bins · labels show bin counts · "
        "bars: descriptive 95% Wilson intervals\n"
        "Training rows (Kimi's distillation inputs), not held-out; "
        "intervals assume independent samples",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout()
    for suffix in ("png", "svg"):
        save_figure(fig, f"figures/qwen38_27b_kimi_training_calibration.{suffix}")
    (OUT / "training_calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    for scope in ("all", "matched_640"):
        print(
            scope,
            {
                t: {k: round(v, 4) for k, v in s.items() if isinstance(v, float)}
                | {"ece_bins": s["ece_bin_sensitivity"]}
                for t, s in report[scope].items()
            },
        )


if __name__ == "__main__":
    main()
