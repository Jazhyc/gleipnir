"""Compare a full-stack C1 sweep with the 1M baseline (latency, throughput, scores)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

SERIES = (("baseline", "#2a78d6", "o"), ("current", "#eb6834", "s"))


def measurements(run: Path) -> dict:
    report = json.loads((run / "summary.json").read_text())
    if report["status"] != "complete":
        raise ValueError(f"{run} is not complete")
    return {
        (phase["phase"], m["prompt_tokens"]): m
        for phase in report["phases"]
        for m in phase["measurements"]
    }


def scores(run: Path, phase: str, length: int) -> list[float]:
    return [
        json.loads((run / phase / f"{length}_repeat{r}.json").read_text())["values"][0][
            "score"
        ]
        for r in range(5)
    ]


def compare(current: Path, baseline: Path, below_8k: Path) -> dict:
    now, base = measurements(current), measurements(baseline)
    rows = []
    for (phase, length), m in sorted(
        now.items(), key=lambda x: (x[0][0] != "native", x[0][1])
    ):
        b = base[(phase, length)]
        reference = baseline if length >= 8192 else below_8k
        rows.append(
            {
                "phase": phase,
                "prompt_tokens": length,
                "baseline_latency_ms": 1000 * b["median_latency_seconds"],
                "current_latency_ms": 1000 * m["median_latency_seconds"],
                "latency_change": m["median_latency_seconds"]
                / b["median_latency_seconds"]
                - 1,
                "baseline_tokens_per_second": b["prompt_tokens_per_second"],
                "current_tokens_per_second": m["prompt_tokens_per_second"],
                "baseline_sm_clock_mhz": b["gpu_samples"].get("sm_clock_median_mhz"),
                "current_sm_clock_mhz": m["gpu_samples"].get("sm_clock_median_mhz"),
                "scores_reference": str(reference),
                "scores_bitwise_equal": scores(current, phase, length)
                == scores(reference, phase, length),
            }
        )
    return {
        "current": str(current),
        "baseline": str(baseline),
        "below_8k_reference": str(below_8k),
        "rows": rows,
        "source_sha256": {
            str(p): hashlib.sha256((p / "summary.json").read_bytes()).hexdigest()
            for p in (current, baseline, below_8k)
        },
    }


def label(tokens: int) -> str:
    return f"{tokens // 1048576}M" if tokens >= 1048576 else f"{tokens // 1024}K"


def plot(result: dict, out: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/gleipnir-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gleipnir.analysis.plotting import set_plot_style

    set_plot_style()
    # Native RoPE through 256K, then the YaRN server beyond the checkpoint limit.
    rows = [r for r in result["rows"] if r["phase"] == "native"] + [
        r
        for r in result["rows"]
        if r["phase"] == "yarn" and r["prompt_tokens"] > 262144
    ]
    xs = [r["prompt_tokens"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), layout="constrained")
    for key, color, marker in SERIES:
        name = "Original FP8 default" if key == "baseline" else "Adopted stack"
        axes[0].plot(
            xs,
            [r[f"{key}_tokens_per_second"] / 1000 for r in rows],
            color=color,
            marker=marker,
            linewidth=2,
            markersize=7,
            label=name,
        )
    axes[1].plot(
        xs,
        [-100 * r["latency_change"] for r in rows],
        color="#eb6834",
        marker="s",
        linewidth=2,
        markersize=7,
    )
    axes[1].axhline(0, color="#898781", linewidth=1)
    for ax, ylabel, title in (
        (axes[0], "Input tokens/s (thousands)", "C1 input throughput"),
        (axes[1], "Latency reduction vs baseline (%)", "C1 latency reduction"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs[::2], [label(x) for x in xs[::2]])
        ax.set_xlabel("Exact input tokens, concurrency 1")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=11)
    fig.suptitle(
        "Gleipnir 4B on one B200: adopted serving stack versus original default"
    )
    for suffix in ("png", "svg"):
        fig.savefig(
            out / f"comparison.{suffix}",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current", type=Path)
    parser.add_argument(
        "--baseline", type=Path, default=Path("results/b200_context_million/million01")
    )
    parser.add_argument(
        "--below-8k", type=Path, default=Path("results/b200_gdn_fast/full02")
    )
    args = parser.parse_args()
    result = compare(args.current, args.baseline, args.below_8k)
    (args.current / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(result, args.current)
    for r in result["rows"]:
        print(
            f"{r['phase']:6} {r['prompt_tokens']:>8} "
            f"{r['baseline_latency_ms']:10.2f} -> "
            f"{r['current_latency_ms']:10.2f} ms ({r['latency_change']:+.1%}) "
            f"exact={r['scores_bitwise_equal']}"
        )


if __name__ == "__main__":
    main()
