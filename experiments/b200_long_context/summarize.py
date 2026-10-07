"""Export complete or partial C1 context measurements and standalone figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

GIB = 1024**3


def records(report: dict) -> list[dict]:
    """Require measured outcomes; never replace failed contexts with zero speed."""
    if report["status"] not in {"complete", "failed"}:
        raise ValueError("long-context run is still active")
    rows = []
    for record in report["measurements"]:
        if record["status"] != "complete" or record["median_latency_seconds"] <= 0:
            raise ValueError("invalid measured context")
        rows.append(
            {
                "phase": record["phase"],
                "prompt_tokens": record["prompt_tokens"],
                "latency_ms": 1000 * record["median_latency_seconds"],
                "latency_min_ms": 1000 * record["min_latency_seconds"],
                "latency_max_ms": 1000 * record["max_latency_seconds"],
                "input_tokens_per_second": record["prompt_tokens_per_second"],
                "requests_per_second": record["requests_per_second"],
                "peak_allocated_gib": record.get("peak_allocated_bytes", 0) / GIB
                if "peak_allocated_bytes" in record
                else None,
                "peak_reserved_gib": record.get("peak_reserved_bytes", 0) / GIB
                if "peak_reserved_bytes" in record
                else None,
                "peak_incremental_allocated_gib": record.get(
                    "peak_incremental_allocated_bytes", 0
                )
                / GIB
                if "peak_incremental_allocated_bytes" in record
                else None,
            }
        )
    return rows


def summarize(directory: Path) -> list[dict]:
    raw = directory / "summary.json"
    report = json.loads(raw.read_text())
    rows = records(report)
    summary = {
        "status": report["status"],
        "rows": rows,
        "error": report.get("error"),
        "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
    for phase, label, color in (
        ("original", "Original 32K envelope", "#D55E00"),
        ("extended", "Extended 256K envelope", "#0072B2"),
    ):
        series = [r for r in rows if r["phase"] == phase]
        if not series:
            continue
        xs = [r["prompt_tokens"] / 1024 for r in series]
        axes[0].plot(
            xs, [r["latency_ms"] for r in series], "o-", color=color, label=label
        )
        axes[0].fill_between(
            xs,
            [r["latency_min_ms"] for r in series],
            [r["latency_max_ms"] for r in series],
            color=color,
            alpha=0.15,
        )
        axes[1].plot(
            xs, [r["input_tokens_per_second"] / 1000 for r in series], "o-", color=color
        )
        if phase == "extended":
            axes[2].plot(
                xs,
                [r["peak_allocated_gib"] for r in series],
                "o-",
                label="Allocated",
                color=color,
            )
            axes[2].plot(
                xs,
                [r["peak_reserved_gib"] for r in series],
                "s--",
                label="Reserved",
                color="#009E73",
            )
    for ax, ylabel in zip(
        axes,
        (
            "Request latency (ms)",
            "Input tokens/s (thousands)",
            "Peak PyTorch memory (GiB)",
        ),
        strict=True,
    ):
        ax.set_xscale("log", base=2)
        ax.set_xticks(
            [8, 16, 32, 64, 128, 256], ["8K", "16K", "32K", "64K", "128K", "256K"]
        )
        ax.set_xlabel("Exact input tokens")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.suptitle("B200 selected arithmetic: single-request context scaling")
    for suffix in ("png", "svg"):
        fig.savefig(
            directory / f"scaling.{suffix}",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    for row in rows:
        print(json.dumps(row))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summarize(args.directory)


if __name__ == "__main__":
    main()
