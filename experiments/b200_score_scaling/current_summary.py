"""Export current-host scaling and explicitly distinct historical controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from experiments.b200_score_scaling.current import CONCURRENCIES, ROOT


def aggregate(report: dict, levels: tuple[int, ...] = CONCURRENCIES) -> list[dict]:
    """Retain all three repeats and reject missing, duplicate or mixed cohorts."""
    if report["status"] != "complete":
        raise ValueError("scaling suite incomplete")
    output = []
    for c in levels:
        trials = [r for r in report["trials"] if r["concurrency"] == c]
        if (
            len(trials) != 3
            or {r["repeat"] for r in trials} != {0, 1, 2}
            or any(r["latency"]["rows"] != 320 for r in trials)
        ):
            raise ValueError("scaling repeat/cohort contract changed")
        row = {"concurrency": c}
        for key in (
            "prompt_tokens_per_second",
            "requests_per_second",
            "p50_seconds",
            "p95_seconds",
            "p99_seconds",
        ):
            values = [r[key] if key in r else r["latency"][key] for r in trials]
            row[key] = statistics.median(values)
            row[key + "_min"] = min(values)
            row[key + "_max"] = max(values)
        output.append(row)
    return output


def summarize(directory: Path) -> dict:
    """Compare fixed-cohort timing; keep host and recipe attribution separate."""
    raw = directory / "summary.json"
    rows = aggregate(json.loads(raw.read_text()))
    old_path = ROOT / "results/b200_score_scaling/scale01/summary.json"
    addendum_path = ROOT / "results/b200_context_scaling/twok01/summary.json"
    old = json.loads(old_path.read_text())
    old["trials"] += json.loads(addendum_path.read_text())["control_trials"]
    historical = aggregate(old)
    fp8_path = ROOT / "results/b200_attention_precision/precision01/fp8/summary.json"
    fp8 = json.loads(fp8_path.read_text())
    fp8_trials = [r for r in fp8["trials"] if r["concurrency"] == 128]
    if len(fp8_trials) != 6 or any(r["latency"]["rows"] != 320 for r in fp8_trials):
        raise ValueError("same-recipe historical c128 control changed")
    fp8_point = {"concurrency": 128}
    for key in ("prompt_tokens_per_second", "p50_seconds", "p95_seconds"):
        values = [r[key] if key in r else r["latency"][key] for r in fp8_trials]
        fp8_point[key] = statistics.median(values)
        fp8_point[key + "_min"] = min(values)
        fp8_point[key + "_max"] = max(values)
    for row, previous in zip(rows, historical, strict=True):
        comparison = json.loads(
            (directory / f"c{row['concurrency']}_comparison.json").read_text()
        )
        delta = comparison["ranking"]["auroc_delta"]
        baseline = comparison["ranking"]["baseline_repeat_median_scores"]
        candidate = comparison["ranking"]["candidate_repeat_median_scores"]
        row.update(
            pooled_auroc_delta_pp=100 * delta["pooled"],
            macro_auroc_delta_pp=100 * delta["macro"],
            pooled_pauroc_at_20_delta_pp=100
            * (
                candidate["pooled"]["pauroc_at_20"] - baseline["pooled"]["pauroc_at_20"]
            ),
            macro_pauroc_at_20_delta_pp=100
            * (
                candidate["macro"]["macro"]["pauroc_at_20"]
                - baseline["macro"]["macro"]["pauroc_at_20"]
            ),
            score_mae=comparison["scores"]["score"]["mean_absolute_difference"],
            threshold_flips=comparison["scores"]["threshold_flips"],
            throughput_ratio_vs_old_recipe=row["prompt_tokens_per_second"]
            / previous["prompt_tokens_per_second"],
            p95_ratio_vs_old_recipe=row["p95_seconds"] / previous["p95_seconds"],
        )
    peak = max(r["prompt_tokens_per_second"] for r in rows)
    summary = {
        "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "rows": rows,
        "historical_024_fp4_curve": historical,
        "historical_031_fp8_c128": fp8_point,
        "historical_receipt_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (old_path, addendum_path, fp8_path)
        },
        "smallest_concurrency_at_95_percent_of_peak": next(
            r["concurrency"]
            for r in rows
            if r["prompt_tokens_per_second"] >= 0.95 * peak
        ),
        "c128_throughput_ratio_vs_same_recipe": rows[-1]["prompt_tokens_per_second"]
        / fp8_point["prompt_tokens_per_second"],
        "interpretation": (
            "Same full320 cohort; old full curve differs in host, vLLM version "
            "and attention projection precision. Same-recipe comparison "
            "available only at c128."
        ),
        "promoted": False,
    }
    (directory / "scaling.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (directory / "scaling.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    from gleipnir.analysis.plotting import set_plot_style

    set_plot_style()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    xs = list(CONCURRENCIES)
    for values, color, label in (
        (rows, "#0072B2", "NC2 · 0.31 FP8"),
        (historical, "#777777", "EU · 0.24 FP4"),
    ):
        axes[0].plot(
            xs,
            [r["prompt_tokens_per_second"] / 1000 for r in values],
            "o-",
            color=color,
            label=label,
        )
        axes[0].fill_between(
            xs,
            [r["prompt_tokens_per_second_min"] / 1000 for r in values],
            [r["prompt_tokens_per_second_max"] / 1000 for r in values],
            color=color,
            alpha=0.15,
        )
    axes[0].scatter(
        [128],
        [fp8_point["prompt_tokens_per_second"] / 1000],
        marker="*",
        s=100,
        color="#009E73",
        label="EU · 0.31 FP8 (c128)",
    )
    axes[0].errorbar(
        [128],
        [fp8_point["prompt_tokens_per_second"] / 1000],
        yerr=[
            [
                (
                    fp8_point["prompt_tokens_per_second"]
                    - fp8_point["prompt_tokens_per_second_min"]
                )
                / 1000
            ],
            [
                (
                    fp8_point["prompt_tokens_per_second_max"]
                    - fp8_point["prompt_tokens_per_second"]
                )
                / 1000
            ],
        ],
        fmt="none",
        color="#009E73",
        capsize=4,
    )
    for key, label, color in (
        ("p50_seconds", "p50", "#0072B2"),
        ("p95_seconds", "p95", "#D55E00"),
    ):
        axes[1].plot(
            xs, [r[key] * 1000 for r in rows], "o-", color=color, label=f"NC2 {label}"
        )
        axes[1].plot(
            xs,
            [r[key] * 1000 for r in historical],
            "--",
            color=color,
            alpha=0.55,
            label=f"EU 0.24 {label}",
        )
        axes[1].fill_between(
            xs,
            [r[key + "_min"] * 1000 for r in rows],
            [r[key + "_max"] * 1000 for r in rows],
            color=color,
            alpha=0.12,
        )
    axes[0].set_ylabel("Input tokens/s (thousands)")
    axes[1].set_ylabel("Request latency (ms)")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(xs, [str(c) for c in xs])
        axis.set_xlabel("Client concurrency")
        axis.set_ylim(bottom=0)
        axis.legend(fontsize=8)
    figure.suptitle("B200 scaling · same 320 prompts · median and repeat range")
    for suffix in ("png", "svg"):
        figure.savefig(directory / f"scaling.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    summary = summarize(parser.parse_args().directory)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
