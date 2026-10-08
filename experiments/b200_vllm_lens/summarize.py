"""Export the frozen Lens overhead comparison without conflating its cohorts."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from experiments.b200_vllm_lens.start import ROOT


def summarize(out: Path) -> dict:
    if json.loads((out / "summary.json").read_text())["status"] != "complete":
        raise ValueError("Lens benchmark incomplete")
    previous = (
        json.loads((out / "reused_compiled_baseline.json").read_text())["path"]
        if (out / "reused_compiled_baseline.json").exists()
        else str(out)
    )
    compiled = {
        1: json.loads((ROOT / previous / "compiled/plain/c1_timing.json").read_text()),
        128: [
            r
            for r in json.loads(
                (
                    ROOT / "results/b200_score_scaling/nc2_fp8_04/summary.json"
                ).read_text()
            )["trials"]
            if r["concurrency"] == 128
        ],
    }
    rows = []
    for condition in ["compiled", "plain", "capture"]:
        for c in [1, 128]:
            timings = (
                compiled[c]
                if condition == "compiled"
                else json.loads((out / condition / f"c{c}_timing.json").read_text())
            )
            if (
                len(timings) != 3
                or {r["repeat"] for r in timings} != {0, 1, 2}
                or any(r["latency"]["rows"] != (64 if c == 1 else 320) for r in timings)
            ):
                raise ValueError("Lens timing/cohort contract changed")
            row = {
                "condition": condition,
                "concurrency": c,
                "workload_rows": 64 if c == 1 else 320,
            }
            for key in [
                "prompt_tokens_per_second",
                "requests_per_second",
                "p50_seconds",
                "p95_seconds",
                "p99_seconds",
            ]:
                values = [r[key] if key in r else r["latency"][key] for r in timings]
                row[key] = statistics.median(values)
                row[key + "_min"], row[key + "_max"] = min(values), max(values)
            if condition != "compiled":
                q = json.loads(
                    (out / condition / f"c{c}_compiled_comparison.json").read_text()
                )
                rank = q["ranking"]
                baseline, candidate = (
                    rank["baseline_repeat_median_scores"],
                    rank["candidate_repeat_median_scores"],
                )
                row.update(
                    score_mae_vs_compiled=q["scores"]["score"][
                        "mean_absolute_difference"
                    ],
                    pooled_auroc_delta_pp=100 * rank["auroc_delta"]["pooled"],
                    macro_auroc_delta_pp=100 * rank["auroc_delta"]["macro"],
                    pooled_pauroc_at_20_delta_pp=100
                    * (
                        candidate["pooled"]["pauroc_at_20"]
                        - baseline["pooled"]["pauroc_at_20"]
                    ),
                    macro_pauroc_at_20_delta_pp=100
                    * (
                        candidate["macro"]["macro"]["pauroc_at_20"]
                        - baseline["macro"]["macro"]["pauroc_at_20"]
                    ),
                )
            rows.append(row)
    result = {
        "rows": rows,
        "analysis_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "benchmark_summary_sha256": hashlib.sha256(
            (out / "summary.json").read_bytes()
        ).hexdigest(),
        "interpretation": (
            "quick64/c1 and full320/c128 are distinct cohorts. Capture changes "
            "HTTP completion/RPC timing and batch composition; this is closed-loop "
            "system overhead, not isolated hook execution cost."
        ),
        "promoted": False,
    }
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "analysis.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    from gleipnir.analysis.plotting import set_plot_style

    set_plot_style()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    colors = ["#777777", "#0072B2", "#009E73"]
    for axis, c, title in zip(
        axes,
        [1, 128],
        ["quick64 · concurrency 1", "full320 · concurrency 128"],
        strict=True,
    ):
        values = [r for r in rows if r["concurrency"] == c]
        med = [r["prompt_tokens_per_second"] / 1000 for r in values]
        axis.bar([0, 1, 2], med, color=colors)
        axis.errorbar(
            [0, 1, 2],
            med,
            yerr=[
                [
                    r["prompt_tokens_per_second"] / 1000
                    - r["prompt_tokens_per_second_min"] / 1000
                    for r in values
                ],
                [
                    r["prompt_tokens_per_second_max"] / 1000
                    - r["prompt_tokens_per_second"] / 1000
                    for r in values
                ],
            ],
            fmt="none",
            color="black",
            capsize=5,
        )
        axis.set_xticks(
            [0, 1, 2], ["Compiled", "Plain eager", "Eager + capture"], fontsize=11
        )
        axis.set_title(title, fontsize=15)
        axis.set_ylabel("Input tokens/s (thousands)")
    fig.suptitle(
        "Monitor Lens · layer 31, last token · median and repeat range", fontsize=16
    )
    for suffix in ["png", "svg"]:
        fig.savefig(out / f"overhead.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(summarize(parser.parse_args().directory), indent=2))


if __name__ == "__main__":
    main()
