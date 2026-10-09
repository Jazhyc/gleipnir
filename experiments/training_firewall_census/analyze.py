"""Join concept logits to fixed-axis changes without treating flags as truth."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from gleipnir.data.monitoring import file_hash, read_rows, write_json, write_rows
from gleipnir.evaluation.direction_stats import correlation, distribution

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/training_firewall_census/config.json"


def join_predictions(predictions: list[dict], paired: list[dict]) -> list[dict]:
    """Require one concept score per original and exact evidence/label identity."""
    by_id = {r["id"]: r for r in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r["index"] for r in paired}:
        raise ValueError("concept/activation membership mismatch")
    joined = []
    for row in paired:
        prediction = by_id[row["index"]]
        for field in ["trajectory_sha256", "student_prompt_sha256", "label"]:
            if prediction[field] != row[field]:
                raise ValueError(f"concept/activation identity mismatch: {field}")
        if row["view"] != "original":
            raise ValueError("non-original activation record")
        if not all(math.isfinite(prediction[k]) for k in ["score", "log_odds"]):
            raise ValueError("nonfinite concept score")
        joined.append(
            row
            | {
                "firewall_score": prediction["score"],
                "firewall_log_odds": prediction["log_odds"],
            }
        )
    return joined


def describe(rows: list[dict]) -> dict:
    """Report continuous associations, threshold groups and fixed rank bands."""
    result = {"rows": len(rows)}
    if not rows:
        return result
    score = [r["firewall_score"] for r in rows]
    log_odds = [r["firewall_log_odds"] for r in rows]
    result["unique_scores"] = len(set(score))
    result["unique_log_odds"] = len(set(log_odds))
    result["score_distribution"] = distribution(score)
    result["log_odds_distribution"] = distribution(log_odds)
    result["spearman_with_log_odds"] = {
        field: correlation(log_odds, [r[field] for r in rows])
        for field in [
            "delta_z20",
            "delta_cos20",
            "trained_z20",
            "soft_target",
            "prompt_tokens",
        ]
    }
    result["thresholds"] = {}
    for threshold in [0.1, 0.5, 0.9]:
        groups = {}
        for positive in [False, True]:
            group = [r for r in rows if (r["firewall_score"] >= threshold) == positive]
            groups["flagged" if positive else "unflagged"] = {
                "rows": len(group),
                **(
                    {"delta_z20": distribution([r["delta_z20"] for r in group])}
                    if group
                    else {}
                ),
            }
        result["thresholds"][str(threshold)] = groups
    ranked = sorted(rows, key=lambda r: (-r["delta_z20"], r["index"]))
    result["top_delta_bands"] = {
        str(fraction): {
            "rows": math.ceil(fraction * len(rows)),
            "flagged": {
                str(t): sum(
                    r["firewall_score"] >= t
                    for r in ranked[: math.ceil(fraction * len(rows))]
                )
                for t in [0.1, 0.5, 0.9]
            },
        }
        for fraction in [0.01, 0.05, 0.1]
    }
    return result


def analyze() -> None:
    config = json.loads(CONFIG.read_text())
    out = ROOT / config["output"]
    paired_path = ROOT / config["activation_input"]
    if file_hash(paired_path) != config["activation_input_sha256"]:
        raise ValueError("activation census hash drift")
    joined = join_predictions(
        read_rows(out / "predictions.jsonl"), read_rows(paired_path)
    )
    if len(joined) != config["expected_rows"]:
        raise ValueError("incomplete population")
    groups = {}
    for source in ["all", *sorted({r["source"] for r in joined})]:
        for label in ["all", 0, 1]:
            for teacher in ["all", "negative", "positive"]:
                selected = [
                    r
                    for r in joined
                    if (source == "all" or r["source"] == source)
                    and (label == "all" or r["label"] == label)
                    and (
                        teacher == "all"
                        or (r["soft_target"] >= 0.5) == (teacher == "positive")
                    )
                ]
                groups[f"{source}/label={label}/teacher={teacher}"] = describe(selected)
    unique = list({r["trajectory_sha256"]: r for r in reversed(joined)}.values())
    write_rows(out / "joined_activations.jsonl", joined)
    candidates = [r for r in joined if r["label"] == 0 and r["soft_target"] >= 0.5]
    for threshold in [0.1, 0.5]:
        counterexamples = sorted(
            [r for r in candidates if r["firewall_score"] < threshold],
            key=lambda r: (-r["delta_z20"], r["index"]),
        )
        write_rows(
            out / f"harmless_teacher_positive_low_firewall_{threshold}.jsonl",
            counterexamples,
        )
    write_json(
        out / "activation_associations.json",
        {
            "groups": groups,
            "unique_trajectory_sensitivity": describe(unique),
            "input_sha256": {
                "activations": file_hash(paired_path),
                "predictions": file_hash(out / "predictions.jsonl"),
                "config": file_hash(CONFIG),
                "analysis_script": file_hash(Path(__file__)),
            },
            "qualification": (
                "Uncalibrated concept-model flags, not verified absence/presence "
                "or causal influence. Pooled associations may reflect source, labels, "
                "targets and length. Original duplicate exposure weights retained; "
                "sensitivity uses first row per exact trajectory."
            ),
        },
    )
    plot(joined, out)
    print(
        json.dumps(
            {
                k: groups[k]
                for k in ["all/label=all/teacher=all", "all/label=0/teacher=positive"]
            }
        )
    )


def plot(rows: list[dict], out: Path) -> None:
    """Compare all harmless records with consistent source and teacher controls."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gleipnir.analysis.plotting import set_plot_style

    set_plot_style()
    harmless = [r for r in rows if r["label"] == 0]
    bins = np.linspace(
        min(r["delta_z20"] for r in harmless), max(r["delta_z20"] for r in harmless), 51
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    for col, teacher_positive in enumerate([False, True]):
        group = [r for r in harmless if (r["soft_target"] >= 0.5) == teacher_positive]
        ax = axes[0, col]
        ax.scatter(
            [r["firewall_log_odds"] for r in group],
            [r["delta_z20"] for r in group],
            s=12,
            alpha=0.35,
        )
        ax.axvline(0, color="gray", ls="--")
        ax.axhline(0, color="gray", ls="--")
        ax.set(
            title=(
                f"Harmless / Kimi target {'≥' if teacher_positive else '<'}0.5 "
                f"(n={len(group):,})"
            ),
            xlabel="Qwen firewall log odds (0 = score 0.5)",
            ylabel="Alignment change (trained − base)",
        )
        ax = axes[1, col]
        for positive, color in [(False, "#0072B2"), (True, "#D55E00")]:
            values = [
                r["delta_z20"]
                for r in group
                if (r["firewall_score"] >= 0.5) == positive
            ]
            if values:
                counts, _ = np.histogram(values, bins=bins)
                assert counts.sum() == len(values)
                density, _ = np.histogram(values, bins=bins, density=True)
                ax.stairs(
                    density,
                    bins,
                    color=color,
                    lw=2,
                    label=f"Qwen {'≥' if positive else '<'}0.5 (n={len(values):,})",
                )
        ax.axvline(0, color="gray", ls="--")
        ax.set(
            xlabel="Alignment change (trained − base)",
            ylabel="Density within concept-score group",
        )
        ax.legend(fontsize=10)
    xlim = (
        min(r["firewall_log_odds"] for r in harmless),
        max(r["firewall_log_odds"] for r in harmless),
    )
    ylim = (bins[0], bins[-1])
    for ax in axes[0]:
        ax.set(xlim=xlim, ylim=ylim)
    max_density = max(ax.get_ylim()[1] for ax in axes[1])
    for ax in axes[1]:
        ax.set(xlim=ylim, ylim=(0, max_density))
    fig.suptitle(
        "Firewall-related content and fixed-direction alignment change", fontsize=16
    )
    fig.supxlabel(
        "Original training inputs; concept presence includes benign discussion. "
        "Model flags are not verified labels.",
        fontsize=10,
    )
    for suffix in ["png", "pdf"]:
        fig.savefig(
            out / f"firewall_activation_distributions.{suffix}",
            dpi=170,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    analyze()
