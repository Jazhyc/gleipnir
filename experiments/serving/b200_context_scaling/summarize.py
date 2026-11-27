"""Export 2K scaling and its contextual full320 comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path

CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128)


def aggregate(trials: list[dict], levels=CONCURRENCIES) -> list[dict]:
    """Retain median and ranges; reject incomplete or mixed-size cohorts."""
    rows = []
    for concurrency in levels:
        group = [t for t in trials if t["concurrency"] == concurrency]
        if len(group) != 3 or any(t["latency"]["rows"] != 320 for t in group):
            raise ValueError("incomplete repeat/workload contract")
        row = {"concurrency": concurrency}
        for key in (
            "prompt_tokens_per_second",
            "requests_per_second",
            "p50_seconds",
            "p95_seconds",
            "p99_seconds",
        ):
            values = [t[key] if key in t else t["latency"][key] for t in group]
            row[key] = statistics.median(values)
            row[key + "_min"], row[key + "_max"] = min(values), max(values)
        rows.append(row)
    return rows


def summarize(directory: Path, prior_directory: Path) -> None:
    raw = directory / "summary.json"
    prior_raw = prior_directory / "summary.json"
    report, prior = json.loads(raw.read_text()), json.loads(prior_raw.read_text())
    if report["status"] != "complete" or prior["status"] != "complete":
        raise ValueError("timing suite incomplete")
    rows = aggregate(report["trials"])
    controls = aggregate(prior["trials"] + report["control_trials"])
    peak = max(r["prompt_tokens_per_second"] for r in rows)
    summary = {
        "rows": rows,
        "contextual_full320": controls,
        "smallest_concurrency_at_95_percent_of_peak": next(
            r["concurrency"]
            for r in rows
            if r["prompt_tokens_per_second"] >= 0.95 * peak
        ),
        "input_tokens_per_prompt": 2048,
        "selection": "unchanged",
        "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "prior_sha256": hashlib.sha256(prior_raw.read_bytes()).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    (directory / "scaling.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (directory / "scaling.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "executed_sources/summarize.py").write_bytes(
        Path(__file__).read_bytes()
    )
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/gleipnir-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gleipnir.analysis.plotting import set_plot_style

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for series, label, color in (
        (rows, "Fixed 2,048 tokens", "#0072B2"),
        (controls, "Original mixed lengths (mean 4,096)", "#D55E00"),
    ):
        xs = [r["concurrency"] for r in series]
        for ax, key, factor in zip(
            axes,
            ("prompt_tokens_per_second", "requests_per_second", "p95_seconds"),
            (0.001, 1, 1000),
            strict=True,
        ):
            ax.plot(
                xs, [r[key] * factor for r in series], "o-", color=color, label=label
            )
            ax.fill_between(
                xs,
                [r[key + "_min"] * factor for r in series],
                [r[key + "_max"] * factor for r in series],
                color=color,
                alpha=0.15,
            )
    for ax, ylabel in zip(
        axes,
        ("Input tokens/s (thousands)", "Requests/s", "p95 latency (ms)"),
        strict=True,
    ):
        ax.set_xscale("log", base=2)
        ax.set_xticks(CONCURRENCIES, [str(x) for x in CONCURRENCIES])
        ax.set_xlabel("Client concurrency")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    fig.suptitle("B200 selected reference: context geometry and concurrency")
    for suffix in ("png", "svg"):
        fig.savefig(
            directory / f"scaling.{suffix}",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("prior_directory", type=Path)
    args = parser.parse_args()
    summarize(args.directory, args.prior_directory)


if __name__ == "__main__":
    main()
