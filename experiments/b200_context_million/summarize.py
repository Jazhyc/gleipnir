"""Export C1 context measurements, the chunk cost fit and standalone figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

GIB = 1024**3
CHUNK = 32768
SERIES = {
    "native": ("FP8 default, checkpoint RoPE", "#2a78d6", "o"),
    "yarn": ("FP8 default, YaRN x4", "#eb6834", "s"),
    "historical": ("0.24 FP4 stack (long02)", "#898781", "^"),
}


def records(directory: Path) -> list[dict]:
    """Require completed measurements; never fill failed lengths."""
    report = json.loads((directory / "summary.json").read_text())
    if report["status"] != "complete":
        raise ValueError("context run is not complete")
    rows = []
    for phase in report["phases"]:
        for m in phase["measurements"]:
            if m["status"] != "complete" or m["median_latency_seconds"] <= 0:
                raise ValueError("invalid measured context")
            samples = m["gpu_samples"]
            rows.append(
                {
                    "series": phase["phase"],
                    "prompt_tokens": m["prompt_tokens"],
                    "latency_ms": 1000 * m["median_latency_seconds"],
                    "latency_min_ms": 1000 * m["min_latency_seconds"],
                    "latency_max_ms": 1000 * m["max_latency_seconds"],
                    "input_tokens_per_second": m["prompt_tokens_per_second"],
                    "input_tokens_per_second_min": m["throughput_min"],
                    "input_tokens_per_second_max": m["throughput_max"],
                    "requests_per_second": m["requests_per_second"],
                    "peak_allocated_gib": m["peak_allocated_bytes"] / GIB,
                    "peak_reserved_gib": m["peak_reserved_bytes"] / GIB,
                    "peak_incremental_allocated_gib": m[
                        "peak_incremental_allocated_bytes"
                    ]
                    / GIB,
                    "sm_clock_median_mhz": samples.get("sm_clock_median_mhz"),
                    "sm_clock_min_mhz": samples.get("sm_clock_min_mhz"),
                    "power_max_w": samples.get("power_max_w"),
                    "power_cap_events": "0x0000000000000004"
                    in samples.get("event_reasons", []),
                }
            )
    return rows


def historical(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())["rows"]
    return [
        {
            "series": "historical",
            "prompt_tokens": r["prompt_tokens"],
            "latency_ms": r["latency_ms"],
            "latency_min_ms": r["latency_min_ms"],
            "latency_max_ms": r["latency_max_ms"],
            "input_tokens_per_second": r["input_tokens_per_second"],
        }
        for r in rows
        if r["phase"] == "extended"
    ]


def chunk_fit(rows: list[dict]) -> dict:
    """Fit n*a + b*n(n+1)/2 over whole-chunk lengths (n chunks of 32K tokens)."""
    points = sorted(
        (r["prompt_tokens"] // CHUNK, r["latency_ms"])
        for r in rows
        if r["prompt_tokens"] >= CHUNK and r["prompt_tokens"] % CHUNK == 0
    )
    n = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points])
    design = np.c_[n, n * (n + 1) / 2]
    # Relative-error weighting so 32K and 1M constrain the fit equally.
    (a, b), *_ = np.linalg.lstsq(design / y[:, None], np.ones_like(y), rcond=None)
    predicted = design @ [a, b]
    return {
        "per_chunk_constant_ms": float(a),
        "per_history_chunk_ms": float(b),
        "points": [
            {
                "prompt_tokens": int(k * CHUNK),
                "measured_ms": float(m),
                "fitted_ms": float(f),
                "relative_error": float((m - f) / m),
                "attention_share": float(b * k * (k + 1) / 2 / f),
            }
            for k, m, f in zip(n, y, predicted, strict=True)
        ],
    }


def label(tokens: int) -> str:
    return f"{tokens // 1048576}M" if tokens >= 1048576 else f"{tokens // 1024}K"


def plot(directory: Path, rows: list[dict], baseline: list[dict]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/gleipnir-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gleipnir.analysis.plotting import set_plot_style

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), layout="constrained")
    ticks = sorted({r["prompt_tokens"] for r in rows})
    for key in ("historical", "native", "yarn"):
        series = (
            baseline if key == "historical" else [r for r in rows if r["series"] == key]
        )
        name, color, marker = SERIES[key]
        style = dict(
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2,
            linestyle="--" if key == "historical" else "-",
            label=name,
        )
        xs = [r["prompt_tokens"] for r in series]
        axes[0].plot(xs, [r["latency_ms"] / 1000 for r in series], **style)
        axes[1].plot(xs, [r["input_tokens_per_second"] / 1000 for r in series], **style)
        if key != "historical":
            axes[2].plot(xs, [r["sm_clock_median_mhz"] for r in series], **style)
            axes[2].fill_between(
                xs,
                [r["sm_clock_min_mhz"] for r in series],
                [r["sm_clock_median_mhz"] for r in series],
                color=color,
                alpha=0.12,
                linewidth=0,
            )
    longest = max(rows, key=lambda r: r["prompt_tokens"])
    axes[0].annotate(
        f"{longest['latency_ms'] / 1000:.1f} s",
        (longest["prompt_tokens"], longest["latency_ms"] / 1000),
        textcoords="offset points",
        xytext=(-12, 8),
        ha="right",
        fontsize=12,
    )
    axes[0].set_yscale("log")
    axes[2].axhline(1965, color="#898781", linewidth=1, linestyle=":")
    axes[2].annotate(
        "max SM clock",
        (ticks[-1], 1965),
        xytext=(0, -16),
        textcoords="offset points",
        ha="right",
        fontsize=11,
        color="#52514e",
    )
    for ax, ylabel, title in zip(
        axes,
        ("Request latency (s, log)", "Input tokens/s (thousands)", "SM clock (MHz)"),
        ("Latency", "Input throughput", "Median SM clock (band: minimum)"),
        strict=True,
    ):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks[::2], [label(t) for t in ticks[::2]])
        ax.set_xlabel("Exact input tokens, concurrency 1")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14)
    axes[1].set_ylim(bottom=0)
    axes[0].legend(fontsize=11, loc="upper left")
    fig.suptitle("Gleipnir 4B FP8 default on one B200: single-request context scaling")
    for suffix in ("png", "svg"):
        fig.savefig(
            directory / f"scaling.{suffix}",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def summarize(directory: Path, historical_directory: Path) -> dict:
    rows = records(directory)
    baseline_path = historical_directory / "scaling.json"
    baseline = historical(baseline_path)
    native = [r for r in rows if r["series"] == "native"]
    yarn = [r for r in rows if r["series"] == "yarn"]
    # Native RoPE through 256K, then the YaRN server beyond the checkpoint limit.
    stitched = native + [r for r in yarn if r["prompt_tokens"] > 262144]
    summary = {
        "rows": rows,
        "historical_rows": baseline,
        "chunk_fit": chunk_fit(stitched),
        "historical_ratio": {
            str(b["prompt_tokens"]): next(
                r["latency_ms"]
                for r in native
                if r["prompt_tokens"] == b["prompt_tokens"]
            )
            / b["latency_ms"]
            for b in baseline
        },
        "yarn_overlap_ratio": {
            str(y["prompt_tokens"]): y["latency_ms"]
            / next(
                r["latency_ms"]
                for r in native
                if r["prompt_tokens"] == y["prompt_tokens"]
            )
            for y in yarn
            if y["prompt_tokens"] <= 262144
        },
        "source_sha256": hashlib.sha256(
            (directory / "summary.json").read_bytes()
        ).hexdigest(),
        "historical_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
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
    plot(directory, rows, baseline)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--historical",
        type=Path,
        default=Path("results/b200_long_context/long02"),
    )
    args = parser.parse_args()
    summary = summarize(args.directory, args.historical)
    print(json.dumps(summary["chunk_fit"], indent=2))
    print(json.dumps(summary["historical_ratio"], indent=2))
    print(json.dumps(summary["yarn_overlap_ratio"], indent=2))


if __name__ == "__main__":
    main()
