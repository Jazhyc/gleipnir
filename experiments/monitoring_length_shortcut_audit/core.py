"""Pure helpers for the monitoring length-shortcut audit."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def pearson(first: list[float], second: list[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Pearson inputs must have equal length of at least two")
    first_mean = statistics.fmean(first)
    second_mean = statistics.fmean(second)
    covariance = sum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first, second, strict=True)
    )
    first_variance = sum((value - first_mean) ** 2 for value in first)
    second_variance = sum((value - second_mean) ** 2 for value in second)
    if first_variance == 0 or second_variance == 0:
        return None
    return covariance / math.sqrt(first_variance * second_variance)


def percentile(values: list[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("percentile requires values and probability in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def binary_metrics(labels: list[int], scores: list[float]) -> dict[str, Any]:
    """Compute binary ranking and threshold metrics without heavy dependencies."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be nonempty and aligned")
    if set(labels) != {0, 1}:
        raise ValueError("binary metrics require both labels")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("scores must be finite probabilities")
    positives = sum(labels)
    negatives = len(labels) - positives
    points = [(0.0, 0.0)]
    true_positives = 0
    false_positives = 0
    ordered = sorted(zip(scores, labels, strict=True), reverse=True)
    index = 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and ordered[stop][0] == ordered[index][0]:
            stop += 1
        for _, label in ordered[index:stop]:
            true_positives += label
            false_positives += 1 - label
        points.append((false_positives / negatives, true_positives / positives))
        index = stop

    def area(max_fpr: float) -> float:
        total = 0.0
        for (left_x, left_y), (right_x, right_y) in zip(
            points, points[1:], strict=True
        ):
            if left_x >= max_fpr:
                break
            clipped_x = min(right_x, max_fpr)
            if right_x == left_x:
                clipped_y = right_y
            else:
                fraction = (clipped_x - left_x) / (right_x - left_x)
                clipped_y = left_y + fraction * (right_y - left_y)
            total += (clipped_x - left_x) * (left_y + clipped_y) / 2
            if right_x >= max_fpr:
                break
        return total

    predictions = [score >= 0.5 for score in scores]
    recall = sum(prediction and label for prediction, label in zip(
        predictions, labels, strict=True
    )) / positives
    fpr = sum(prediction and not label for prediction, label in zip(
        predictions, labels, strict=True
    )) / negatives
    unique_scores = len(set(scores))
    return {
        "n": len(labels),
        "auroc": area(1.0),
        "pauroc_at_20": area(0.2) / 0.2,
        "brier": statistics.fmean(
            (score - label) ** 2
            for score, label in zip(scores, labels, strict=True)
        ),
        "balanced_accuracy": (recall + 1 - fpr) / 2,
        "recall": recall,
        "fpr": fpr,
        "unique_scores": unique_scores,
        "tied_rows": len(scores) - unique_scores,
    }


def grouped_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source"])].append(row)
    sources = {
        source: binary_metrics(
            [int(row["label"]) for row in subset],
            [float(row["score"]) for row in subset],
        )
        for source, subset in sorted(grouped.items())
    }
    metric_names = (
        "auroc",
        "pauroc_at_20",
        "brier",
        "balanced_accuracy",
        "recall",
        "fpr",
    )
    return {
        "sources": sources,
        "macro": {
            name: statistics.fmean(source[name] for source in sources.values())
            for name in metric_names
        },
        "pooled": binary_metrics(
            [int(row["label"]) for row in rows],
            [float(row["score"]) for row in rows],
        ),
    }


def length_association(
    rows: list[dict[str, Any]], *, value_key: str
) -> dict[str, Any]:
    lengths = [float(row["tokens"]) for row in rows]
    labels = [float(row["label"]) for row in rows]
    values = [float(row[value_key]) for row in rows]
    negative = [row["tokens"] for row in rows if int(row["label"]) == 0]
    positive = [row["tokens"] for row in rows if int(row["label"]) == 1]
    length_auc = binary_metrics(
        [int(row["label"]) for row in rows],
        [rank_probability(float(row["tokens"]), lengths) for row in rows],
    )["auroc"]
    return {
        "n": len(rows),
        "negative_mean_tokens": statistics.fmean(negative),
        "positive_mean_tokens": statistics.fmean(positive),
        "negative_median_tokens": statistics.median(negative),
        "positive_median_tokens": statistics.median(positive),
        "length_only_label_auroc": length_auc,
        "length_label_pearson": pearson(lengths, labels),
        f"length_{value_key}_pearson": pearson(lengths, values),
    }


def rank_probability(value: float, reference: list[float]) -> float:
    below = sum(candidate < value for candidate in reference)
    equal = sum(candidate == value for candidate in reference)
    return (below + 0.5 * equal) / len(reference)


def association_by_source(
    rows: list[dict[str, Any]], *, value_key: str
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source"])].append(row)
    return {
        "pooled": length_association(rows, value_key=value_key),
        "sources": {
            source: length_association(subset, value_key=value_key)
            for source, subset in sorted(grouped.items())
        },
    }


def select_length_stratified(
    rows: list[dict[str, Any]],
    *,
    quartiles: int,
    rows_per_stratum: int,
    maximum_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select deterministic source-label-length-balanced rows."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row["metadata"]
        key = str(metadata["source_dataset"]), int(metadata["ground_truth"])
        grouped[key].append(row)
    selected = []
    for (source, label), subset in sorted(grouped.items()):
        ordered = sorted(
            subset,
            key=lambda row: (
                int(row["metadata"]["predicted_provider_prompt_tokens"]),
                str(row["id"]),
            ),
        )
        for quartile in range(quartiles):
            start = len(ordered) * quartile // quartiles
            stop = len(ordered) * (quartile + 1) // quartiles
            eligible = [
                row
                for row in ordered[start:stop]
                if int(row["metadata"]["predicted_provider_prompt_tokens"])
                <= maximum_tokens
            ]
            ranked = sorted(
                eligible,
                key=lambda row: hashlib.sha256(
                    f"{seed}:{row['id']}".encode()
                ).hexdigest(),
            )
            if len(ranked) < rows_per_stratum:
                raise ValueError(
                    f"insufficient rows for {source=} {label=} {quartile=}"
                )
            for row in ranked[:rows_per_stratum]:
                selected.append(
                    {
                        **row,
                        "audit_length_quartile": quartile,
                    }
                )
    return selected


def padded_prompt(prompt: str, *, marker: str, line: str, repetitions: int) -> str:
    if prompt.count(marker) != 1:
        raise ValueError("padding insertion marker must occur exactly once")
    if repetitions == 0:
        return prompt
    block = (
        "<length_control_padding>\n"
        + (line + "\n") * repetitions
        + "</length_control_padding>\n\n"
    )
    return prompt.replace(marker, block + marker)


def materialize_counterfactual_rows(
    selected: list[dict[str, Any]],
    *,
    marker: str,
    line: str,
    conditions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    names = [str(condition["name"]) for condition in conditions]
    if len(names) != len(set(names)) or "original" not in names:
        raise ValueError("conditions must be unique and include original")
    output = []
    for row in selected:
        for condition in conditions:
            name = str(condition["name"])
            prompt = padded_prompt(
                str(row["prompt"]),
                marker=marker,
                line=line,
                repetitions=int(condition["repetitions"]),
            )
            metadata = {
                **row["metadata"],
                "audit_base_id": str(row["id"]),
                "audit_condition": name,
                "audit_length_quartile": int(row["audit_length_quartile"]),
                "audit_padding_repetitions": int(condition["repetitions"]),
                "audit_original_rendered_prompt_sha256": row["metadata"][
                    "rendered_prompt_sha256"
                ],
                "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
            output.append(
                {
                    "id": f"{row['id']}::{name}",
                    "metadata": metadata,
                    "prompt": prompt,
                }
            )
    return output


def length_matched_rows(
    rows: list[dict[str, Any]], *, bins: int = 10
) -> list[dict[str, Any]]:
    """Coarsened-exact-match labels by source and pooled length quantile."""
    if bins < 1:
        raise ValueError("length matching requires at least one bin")
    matched = []
    for source in sorted({str(row["source"]) for row in rows}):
        source_rows = sorted(
            (row for row in rows if str(row["source"]) == source),
            key=lambda row: (int(row["tokens"]), str(row["id"])),
        )
        for bin_index in range(bins):
            start = len(source_rows) * bin_index // bins
            stop = len(source_rows) * (bin_index + 1) // bins
            labels = {
                label: [
                    row
                    for row in source_rows[start:stop]
                    if int(row["label"]) == label
                ]
                for label in (0, 1)
            }
            count = min(len(labels[0]), len(labels[1]))
            for label in (0, 1):
                subset = labels[label]
                indices = [
                    min(
                        len(subset) - 1,
                        int((index + 0.5) * len(subset) / count),
                    )
                    for index in range(count)
                ]
                matched.extend(subset[index] for index in indices)
    return matched


def paired_padding_summary(
    predictions: list[dict[str, Any]],
    *,
    conditions: list[str],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    by_base: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in predictions:
        base, separator, condition = str(row["id"]).rpartition("::")
        if not separator or condition not in conditions:
            raise ValueError(f"unexpected audit prediction id {row['id']!r}")
        by_base[base][condition] = row
    if any(set(rows) != set(conditions) for rows in by_base.values()):
        raise ValueError("counterfactual predictions are incomplete")
    result: dict[str, Any] = {}
    originals = [rows["original"] for rows in by_base.values()]
    for condition in conditions:
        if condition == "original":
            continue
        compared = [rows[condition] for rows in by_base.values()]
        differences = [
            float(right["score"]) - float(left["score"])
            for left, right in zip(originals, compared, strict=True)
        ]
        absolute = [abs(value) for value in differences]
        flips = [
            (float(left["score"]) >= 0.5) != (float(right["score"]) >= 0.5)
            for left, right in zip(originals, compared, strict=True)
        ]
        token_deltas = [
            int(right["prompt_tokens"]) - int(left["prompt_tokens"])
            for left, right in zip(originals, compared, strict=True)
        ]
        summary = {
            "rows": len(differences),
            "mean_signed_score_difference": statistics.fmean(differences),
            "mean_absolute_score_difference": statistics.fmean(absolute),
            "p95_absolute_score_difference": percentile(absolute, 0.95),
            "maximum_absolute_score_difference": max(absolute),
            "threshold_flips": sum(flips),
            "threshold_flip_fraction": statistics.fmean(flips),
            "mean_added_prompt_tokens": statistics.fmean(token_deltas),
            "minimum_added_prompt_tokens": min(token_deltas),
            "maximum_added_prompt_tokens": max(token_deltas),
            "pearson_with_original": pearson(
                [float(row["score"]) for row in originals],
                [float(row["score"]) for row in compared],
            ),
        }
        summary["material"] = bool(
            summary["mean_absolute_score_difference"]
            >= thresholds["mean_absolute_score_difference"]
            or abs(summary["mean_signed_score_difference"])
            >= thresholds["absolute_mean_signed_score_difference"]
            or summary["p95_absolute_score_difference"]
            >= thresholds["p95_absolute_score_difference"]
            or summary["threshold_flip_fraction"]
            >= thresholds["threshold_flip_fraction"]
        )
        result[condition] = summary
    return result
