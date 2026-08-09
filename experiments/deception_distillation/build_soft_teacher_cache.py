#!/usr/bin/env python3
"""Convert audited direct-judge distributions into soft student targets."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def aggregate_soft_targets(
    records: list[dict[str, Any]],
    *,
    expected_members: int = 3,
) -> list[dict[str, Any]]:
    """Select each row's max-score member and normalize logits label-blind."""
    grouped: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("parse_error"):
            raise ValueError("teacher cache contains a parse error")
        if record.get("missing_rating_token_ids"):
            raise ValueError("teacher cache contains missing rating logits")
        probs = {
            int(key): float(value) for key, value in record["rating_probs"].items()
        }
        if abs(sum(probs.values()) - 1.0) > 1e-9:
            raise ValueError("teacher rating probabilities do not normalize to one")
        grouped[(str(record["dataset"]), record["index"])].append(record)
    if not grouped:
        raise ValueError("teacher cache is empty")

    selected: list[dict[str, Any]] = []
    for key, members in grouped.items():
        if len(members) != expected_members:
            raise ValueError(
                f"expected {expected_members} teacher members for {key}, "
                f"got {len(members)}"
            )
        member_names = {str(member["ensemble_member"]) for member in members}
        if len(member_names) != expected_members:
            raise ValueError(f"duplicate teacher member for {key}")
        labels = {int(member["label"]) for member in members}
        if len(labels) != 1:
            raise ValueError(f"teacher labels disagree for {key}")
        winner = max(members, key=lambda member: float(member["score"]))
        score = min(max(float(winner["score"]), 1e-12), 1.0 - 1e-12)
        selected.append(
            {
                "dataset": key[0],
                "index": key[1],
                "label": labels.pop(),
                "teacher_score": score,
                "teacher_logit": math.log(score / (1.0 - score)),
                "selected_member": winner["ensemble_member"],
                "rating_probs": winner["rating_probs"],
            }
        )

    logits = [record["teacher_logit"] for record in selected]
    mean = statistics.mean(logits)
    scale = statistics.pstdev(logits)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"teacher logit scale must be positive, got {scale}")
    for record in selected:
        normalized_logit = (record["teacher_logit"] - mean) / scale
        record["soft_target"] = 1.0 / (1.0 + math.exp(-normalized_logit))
        record["normalization"] = {
            "kind": "global_label_blind_logit_zscore",
            "mean": mean,
            "scale": scale,
        }
    return sorted(
        selected, key=lambda record: (record["dataset"], str(record["index"]))
    )


def build_binary_soft_targets(
    records: list[dict[str, Any]],
    *,
    honest_name: str = "negative",
    deceptive_name: str = "positive",
) -> list[dict[str, Any]]:
    """Preserve a single binary teacher's normalized positive probability."""
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, Any]] = set()
    for record in records:
        key = (str(record["dataset"]), record["index"])
        if key in seen:
            raise ValueError(f"duplicate binary teacher row for {key}")
        seen.add(key)
        if record.get("parse_error"):
            raise ValueError(f"binary teacher cache contains a parse error for {key}")
        if record.get("missing_rating_token_ids"):
            raise ValueError(f"binary teacher cache contains missing logits for {key}")
        probabilities = {
            str(name): float(value) for name, value in record["target_probs"].items()
        }
        if set(probabilities) != {honest_name, deceptive_name}:
            raise ValueError(
                f"binary teacher targets for {key} must be "
                f"{sorted((honest_name, deceptive_name))}, got "
                f"{sorted(probabilities)}"
            )
        if any(
            not math.isfinite(value) or value < 0.0 for value in probabilities.values()
        ):
            raise ValueError(f"invalid binary teacher probability for {key}")
        total = sum(probabilities.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"binary teacher probabilities do not normalize for {key}: {total}"
            )
        soft_target = probabilities[deceptive_name]
        reported_score = float(record["score"])
        if not math.isclose(reported_score, soft_target, abs_tol=1e-12):
            raise ValueError(
                f"binary teacher score disagrees with deceptive probability for {key}"
            )
        clipped = min(max(soft_target, 1e-12), 1.0 - 1e-12)
        selected.append(
            {
                "dataset": key[0],
                "index": key[1],
                "label": int(record["label"]),
                "teacher_score": reported_score,
                "teacher_logit": math.log(clipped / (1.0 - clipped)),
                "target_probs": probabilities,
                "soft_target": soft_target,
                "normalization": {"kind": "binary_identity"},
            }
        )
    if not selected:
        raise ValueError("binary teacher cache is empty")
    return sorted(
        selected, key=lambda record: (record["dataset"], str(record["index"]))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--additional-input",
        action="append",
        default=[],
        type=Path,
        help="Additional disjoint teacher JSONL cache to merge before validation.",
    )
    parser.add_argument("--expected-rows", type=int, default=2880)
    parser.add_argument(
        "--kind",
        choices=("dks_zscore", "binary_identity"),
        default="dks_zscore",
    )
    args = parser.parse_args()

    records = []
    for path in [args.input, *args.additional_input]:
        records.extend(
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    aggregated = (
        build_binary_soft_targets(records)
        if args.kind == "binary_identity"
        else aggregate_soft_targets(records)
    )
    if len(aggregated) != args.expected_rows:
        raise ValueError(
            f"expected {args.expected_rows} aggregated rows, got {len(aggregated)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in aggregated)
    )
    normalization = aggregated[0]["normalization"]
    message = (
        f"wrote {len(aggregated)} rows to {args.output}; "
        f"normalization={normalization['kind']}"
    )
    if normalization["kind"] == "global_label_blind_logit_zscore":
        message += (
            f" teacher_logit_mean={normalization['mean']:.12g}"
            f" teacher_logit_scale={normalization['scale']:.12g}"
        )
    print(message)


if __name__ == "__main__":
    main()
