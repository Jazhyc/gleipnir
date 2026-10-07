"""Export the measured reference scaling table and standalone figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import tempfile
from pathlib import Path

CONCURRENCIES = (1, 2, 4, 16, 32, 64, 128)


def aggregate(report: dict) -> list[dict]:
    """Reject incomplete or mixed-cohort sweeps; retain repeat ranges."""
    if report.get("status") != "complete":
        raise ValueError("scaling suite incomplete")
    rows = []
    for concurrency in CONCURRENCIES:
        trials = [t for t in report["trials"] if t["concurrency"] == concurrency]
        if len(trials) != 3 or any(t["latency"]["rows"] != 320 for t in trials):
            raise ValueError("scaling repeat/workload contract changed")
        result = {"concurrency": concurrency}
        for key in ("prompt_tokens_per_second", "requests_per_second"):
            values = [t[key] for t in trials]
            result[key] = statistics.median(values)
            result[key + "_min"] = min(values)
            result[key + "_max"] = max(values)
        for key in ("p50_seconds", "p95_seconds", "p99_seconds"):
            values = [t["latency"][key] for t in trials]
            result[key] = statistics.median(values)
            result[key + "_min"] = min(values)
            result[key + "_max"] = max(values)
        rows.append(result)
    return rows


def summarize(directory: Path) -> list[dict]:
    """Write table/receipts after hardware measurements; plotting runs on CPU."""
    raw = directory / "summary.json"
    report = json.loads(raw.read_text())
    rows = aggregate(report)
    for row in rows:
        comparison = json.loads(
            (directory / f"c{row['concurrency']}_comparison.json").read_text()
        )
        delta = comparison["ranking"]["auroc_delta"]
        row["pooled_auroc_delta_pp"] = 100 * delta["pooled"]
        row["macro_auroc_delta_pp"] = 100 * delta["macro"]
        row["threshold_flips"] = comparison["scores"]["threshold_flips"]
    with (directory / "scaling.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    peak = max(rows, key=lambda r: r["prompt_tokens_per_second"])
    summary = {
        "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "rows": rows,
        "peak_measured_concurrency": peak["concurrency"],
        "smallest_concurrency_at_95_percent_of_peak": next(
            r["concurrency"]
            for r in rows
            if r["prompt_tokens_per_second"] >= 0.95 * peak["prompt_tokens_per_second"]
        ),
        "workload": "full320_training_seen_systems",
        "selection": "unchanged",
    }
    (directory / "scaling.json").write_text(json.dumps(summary, indent=2) + "\n")
    (directory / "executed_analysis").mkdir(exist_ok=True)
    (directory / "executed_analysis/summarize.py").write_bytes(
        Path(__file__).read_bytes()
    )

    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gleipnir-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gleipnir.analysis.plotting import set_plot_style

    set_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    xs = [r["concurrency"] for r in rows]
    ys = [r["prompt_tokens_per_second"] / 1000 for r in rows]
    axes[0].plot(xs, ys, "o-", color="#0072B2")
    axes[0].fill_between(
        xs,
        [r["prompt_tokens_per_second_min"] / 1000 for r in rows],
        [r["prompt_tokens_per_second_max"] / 1000 for r in rows],
        color="#0072B2",
        alpha=0.18,
    )
    axes[0].set_ylabel("Input tokens/s (thousands)")
    axes[0].set_title("Median; shading shows repeat range")
    axes[0].set_ylim(bottom=0)
    for key, label, color in (
        ("p50_seconds", "p50", "#0072B2"),
        ("p95_seconds", "p95", "#D55E00"),
    ):
        axes[1].plot(xs, [1000 * r[key] for r in rows], "o-", label=label, color=color)
    axes[1].set_ylabel("Request latency (ms)")
    axes[1].legend()
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.set_xlabel("Client concurrency")
    figure.suptitle("B200 selected reference: same 320 prompts at every level")
    for suffix in ("png", "svg"):
        figure.savefig(
            directory / f"scaling.{suffix}",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    for row in summarize(args.directory):
        print(json.dumps(row))


if __name__ == "__main__":
    main()
