"""Complete original-data alignment statistics and descriptive review candidates."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from gleipnir.data.monitoring import read_rows, write_json, write_rows
from gleipnir.evaluation.direction_stats import (
    correlation,
    describe,
    paired_values,
)


def positive_mass_share(values: list[float], total: float) -> float | None:
    """Describe positive scalar mass, without equating it to training influence."""
    return sum(max(0.0, v) for v in values) / total if total > 0 else None


def summarize(rows: list[dict]) -> dict:
    return describe(rows) | {
        "teacher_disagreements": sum(
            (r["soft_target"] >= 0.5) != r["label"] for r in rows
        ),
        "census_flags_05": sum(r["census_score"] >= 0.5 for r in rows),
        "census_flags_09": sum(r["census_score"] >= 0.9 for r in rows),
        "census_associations": {
            field: correlation(
                [r[field] for r in rows], [r["census_score"] for r in rows]
            )
            for field in ["base_z20", "trained_z20", "delta_z20", "delta_cos20"]
        },
        "base_trained_rho": correlation(
            [r["base_z20"] for r in rows], [r["trained_z20"] for r in rows]
        ),
    }


def analyze(out: Path, root: Path, c: dict) -> None:
    base = read_rows(out / "base.jsonl")
    trained = read_rows(out / "trained.jsonl")
    if len(base) != 8688 or len(trained) != 8688:
        raise ValueError("incomplete original population")
    rs = [paired_values(b, t) for b, t in zip(base, trained, strict=True)]
    if len({r["id"] for r in rs}) != 8688 or any(r["view"] != "original" for r in rs):
        raise ValueError("duplicate or synthetic comparison row")
    groups = {"all": rs}
    for field in ["source", "dataset", "label"]:
        for value in sorted({str(r[field]) for r in rs}):
            groups[field + "/" + value] = [r for r in rs if str(r[field]) == value]
    for label in [0, 1]:
        for teacher_positive in [0, 1]:
            groups[f"label_teacher/{label}/{teacher_positive}"] = [
                r
                for r in rs
                if r["label"] == label
                and int(r["soft_target"] >= 0.5) == teacher_positive
            ]
        for src in sorted({r["source"] for r in rs}):
            groups[f"source_label/{src}/{label}"] = [
                r for r in rs if r["source"] == src and r["label"] == label
            ]
    for low, high in zip(c["teacher_bins"][:-1], c["teacher_bins"][1:], strict=True):
        groups[f"teacher_bin/{low}/{high}"] = [
            r
            for r in rs
            if low <= r["soft_target"] < high or high == 1 and r["soft_target"] == 1
        ]
    for threshold in [0.5, 0.9]:
        for positive in [False, True]:
            groups[f"census/{threshold}/{positive}"] = [
                r for r in rs if (r["census_score"] >= threshold) == positive
            ]
    summary = {
        "groups": {name: summarize(rows) for name, rows in groups.items() if rows},
        "scalar_mass_by_source": {},
        "rank_bands": {},
    }
    fields = ["trained_z20", "delta_z20", "delta_cos20", "base_z20"]
    bands = {}
    for field in fields:
        total = sum(max(0.0, r[field]) for r in rs)
        summary["scalar_mass_by_source"][field] = {
            src: {
                "positive_mass_share": positive_mass_share(
                    [r[field] for r in rs if r["source"] == src], total
                ),
                "rows": sum(r["source"] == src for r in rs),
            }
            for src in sorted({r["source"] for r in rs})
        }
        ranks = rankdata([r[field] for r in rs], method="average") / len(rs)
        for r, rank in zip(rs, ranks, strict=True):
            r[field + "_percentile"] = float(rank)
        ordered = sorted(rs, key=lambda r: (-r[field], r["id"]))
        summary["rank_bands"][field] = {}
        for fraction in c["rank_fractions"]:
            selected = ordered[: math.ceil(len(rs) * fraction)]
            bands[field, fraction] = {r["id"] for r in selected}
            summary["rank_bands"][field][str(fraction)] = summarize(selected) | {
                "positive_mass_share": positive_mass_share(
                    [r[field] for r in selected], total
                ),
                "source_counts": {
                    src: sum(r["source"] == src for r in selected)
                    for src in sorted({r["source"] for r in rs})
                },
                "label_counts": {
                    str(label): sum(r["label"] == label for r in selected)
                    for label in [0, 1]
                },
            }
        write_rows(out / (field + "_ranking.jsonl"), ordered)
    originals = {
        (r["dataset"], str(r["index"])): r
        for r in read_rows(root / c["inputs"]["clean"]["path"])
    }

    def excerpt(r):
        text = originals[r["dataset"], r["index"]]["student_prompt"]
        text = text.partition("<agent_trajectory>\n")[2].rsplit(
            "</agent_trajectory>\n", 1
        )[0]
        snippets = []
        for needle in c["excerpt_terms"]:
            i = text.lower().find(needle.lower())
            if i >= 0 and len(snippets) < 4:
                snippets.append(text[max(0, i - 180) : i + 300])
        return r | {
            "trajectory_start": text[:700],
            "trajectory_end": text[-700:],
            "context_excerpts": snippets,
        }

    for field in fields:
        for sign, factor in [("high", -1), ("low", 1)]:
            selected = sorted(rs, key=lambda r: (factor * r[field], r["id"]))[
                : c["excerpt_rows"]
            ]
            write_rows(
                out / (field + "_" + sign + "_examples.jsonl"),
                [excerpt(r) for r in selected],
            )
    flags = []
    for r in rs:
        flags.append(
            {
                k: r[k]
                for k in [
                    "id",
                    "dataset",
                    "index",
                    "source",
                    "label",
                    "lineage_group",
                    "prompt_sha256",
                    "student_prompt_sha256",
                    "trajectory_sha256",
                    "soft_target",
                    "census_score",
                ]
            }
            | {
                "teacher_hard_disagreement": (r["soft_target"] >= 0.5) != r["label"],
                **{
                    field + "_top_" + str(f): r["id"] in bands[field, f]
                    for field in fields
                    for f in c["rank_fractions"]
                },
            }
        )
    write_rows(out / "review_flags.jsonl", flags)
    write_rows(out / "paired_models.jsonl", rs)
    summary["timing"] = {}
    for model, values in [("base", base), ("trained", trained)]:
        batches = [
            json.loads(p.read_text()) for p in (out / "batches" / model).glob("*.json")
        ]
        seconds = sum(r["seconds"] for r in batches)
        tokens = sum(r["tokens"] for r in batches)
        summary["timing"][model] = {
            "seconds": seconds,
            "input_tokens": tokens,
            "input_tokens_per_second": tokens / seconds,
            "requests_per_second": len(rs) / seconds,
            "latency_p50_p95_seconds": np.quantile(
                [r["latency_seconds"] for r in values], [0.5, 0.95]
            ).tolist(),
            "latency_includes_semaphore_wait": True,
        }
    summary["qualification"] = (
        "Complete 8688 original training rows; no added synthetic views. Alignment and "
        "positive scalar-mass shares are not causal training influence. "
        "Census flags are "
        "fallible model judgments. Review flags are descriptive; no data was filtered."
    )
    write_json(out / "summary.json", summary)
