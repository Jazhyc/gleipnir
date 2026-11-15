"""Plot latency against throughput and against prompt length."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"  # portable SVG: text as outlines

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results/serving_concurrency/summary.json"
OUTPUT = ROOT / "figures/serving_latency.pdf"
MODELS = {"4b": ("Gleipnir 4B", "#E45756"), "27b": ("Gleipnir 27B", "#4C78A8")}
STACKS = {"fast": ("Released stack", "-"), "bf16": ("Stock vLLM, BF16", "--")}
LABELLED = (1, 16, 128)  # concurrency levels written next to their points
NATIVE = 262144
SHORTEST = 4096  # shorter prompts take about the same time and are not shown


def sweep(summary: dict, condition: str) -> list[tuple[int, float, float]]:
    """(concurrency, prompt tokens/s, median latency in ms), by concurrency."""
    points = [
        (
            int(key.removeprefix("full_c")),
            item["prompt_tokens_per_second"],
            item["p50_ms"],
        )
        for key, item in summary.get(condition, {}).items()
        if key.startswith("full_c")
    ]
    return sorted(points)


def lengths(summary: dict, condition: str) -> list[tuple[int, float]]:
    """(prompt tokens, median latency in ms) over both phases, by length.

    At 262,144 tokens both phases have a point; the native one is kept.
    """
    merged: dict[int, float] = {}
    for phase in ("yarn", "native"):
        entries = summary.get(f"{condition}_{phase}", {}).get("lengths", {})
        merged.update({int(n): item["median_ms"] for n, item in entries.items()})
    return sorted(item for item in merged.items() if item[0] >= SHORTEST)


def legend(figure: plt.Figure) -> None:
    handles = [
        Line2D([], [], color=color, lw=2, label=name) for name, color in MODELS.values()
    ] + [
        Line2D([], [], color="#444444", lw=1.6, ls=style, label=name)
        for name, style in STACKS.values()
    ]
    figure.legend(
        handles=handles, frameon=False, fontsize=8, ncol=4, loc="upper center"
    )


def plot(summary: dict, output: Path) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for model, (_, color) in MODELS.items():
        for stack, (_, style) in STACKS.items():
            condition = f"{model}_{stack}"
            points = sweep(summary, condition)
            if points:
                left.plot(
                    [p[1] / 1000 for p in points],
                    [p[2] for p in points],
                    color=color,
                    ls=style,
                    lw=1.6,
                    marker="o",
                    ms=3,
                )
                for concurrency, tokens, latency in points:
                    if stack == "fast" and concurrency in LABELLED:
                        left.annotate(
                            str(concurrency),
                            (tokens / 1000, latency),
                            xytext=(4, -9),
                            textcoords="offset points",
                            fontsize=7,
                            color=color,
                        )
            curve = lengths(summary, condition)
            if curve:
                right.plot(
                    [p[0] for p in curve],
                    [p[1] for p in curve],
                    color=color,
                    ls=style,
                    lw=1.6,
                    marker="o",
                    ms=3,
                )
    left.set_xscale("log")
    left.set_xticks([20, 50, 100, 200], labels=["20", "50", "100", "200"])
    left.minorticks_off()
    left.set_yscale("log")
    left.set_xlabel("Prompt tokens per second (thousands)")
    left.set_ylabel("Median latency (ms)")
    left.set_title("Throughput saturates near 16 requests", fontsize=9)
    right.set_xscale("log", base=2)
    right.set_yscale("log")
    right.axvline(NATIVE, color="#999999", lw=0.8, ls=":")
    right.annotate(
        "YaRN",
        (NATIVE, 1),
        xycoords=("data", "axes fraction"),
        xytext=(3, -10),
        textcoords="offset points",
        fontsize=7,
        color="#666666",
    )
    right.set_xlabel("Prompt tokens")
    right.set_ylabel("Latency of one request (ms)")
    right.set_title("About 2× faster on long prompts", fontsize=9)
    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.25, lw=0.5)
        axis.tick_params(labelsize=8)
    legend(figure)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    plot(json.loads(arguments.summary.read_text()), arguments.output)


if __name__ == "__main__":
    main()
