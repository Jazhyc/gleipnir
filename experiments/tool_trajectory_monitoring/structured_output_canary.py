"""Prepare and compare the paired Kimi scalar structured-output canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from experiments.tool_trajectory_monitoring.teacher_canary import (
    atomic_write_json,
    load_jsonl,
    sha256_file,
)

DEFAULT_BASE_INPUT = Path("data/tool_trajectory_monitoring/canary/prompts.jsonl")
DEFAULT_INPUT = Path(
    "data/tool_trajectory_monitoring/structured_output_canary/prompts.jsonl"
)
DEFAULT_MANIFEST = Path(
    "data/tool_trajectory_monitoring/structured_output_canary/prompts.manifest.json"
)
DEFAULT_UNSTRUCTURED = Path(
    "results/tool_trajectory_monitoring/kimi_k3_scalar_unstructured_canary.jsonl"
)
DEFAULT_STRUCTURED = Path(
    "results/tool_trajectory_monitoring/kimi_k3_scalar_structured_canary.jsonl"
)
DEFAULT_SUMMARY = Path(
    "results/tool_trajectory_monitoring/kimi_k3_structured_output_canary_summary.json"
)
DEFAULT_ROWS = 20
SCALAR_INTERFACE_PATH = (
    Path(__file__).resolve().parent / "prompts" / "teacher_scalar_interface.txt"
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_paired_prompts(
    rows: list[dict[str, Any]],
    *,
    total_rows: int = DEFAULT_ROWS,
) -> list[dict[str, Any]]:
    """Replace only the teacher output interface in a balanced canary prefix."""
    if total_rows < 4 or total_rows % 4:
        raise ValueError("total_rows must be a positive multiple of four")
    if len(rows) < total_rows:
        raise ValueError(f"requested {total_rows} rows but only {len(rows)} exist")
    teacher = load_prompt_set().teacher
    marker = "OUTPUT INTERFACE\n"
    if teacher.instruction.count(marker) != 1:
        raise ValueError("teacher instruction must contain one OUTPUT INTERFACE")
    policy, _ = teacher.instruction.split(marker)
    scalar_interface = SCALAR_INTERFACE_PATH.read_text(encoding="utf-8").strip()
    scalar_instruction = (
        f"{policy}{scalar_interface}\n\n"
        "The action-only trajectory follows this instruction.\n"
    )
    scalar_teacher = replace(teacher, instruction=scalar_instruction)
    prepared = []
    for source in rows[:total_rows]:
        prompt = str(source["prompt"])
        if not prompt.startswith(teacher.cache_prefix):
            raise ValueError(f"row {source['id']!r} does not use the teacher prefix")
        decision_prompt = (
            scalar_teacher.cache_prefix + prompt[len(teacher.cache_prefix) :]
        )
        metadata = dict(source.get("metadata") or {})
        metadata["base_rendered_prompt_sha256"] = metadata.get(
            "rendered_prompt_sha256"
        )
        metadata["rendered_prompt_sha256"] = _sha256_text(decision_prompt)
        metadata["output_interface"] = "binary-json-scalar-v1"
        metadata["cache_prefix_chars"] = len(scalar_teacher.cache_prefix)
        metadata["cache_prefix_sha256"] = _sha256_text(scalar_teacher.cache_prefix)
        prepared.append(
            {
                "id": str(source["id"]),
                "prompt": decision_prompt,
                "metadata": metadata,
            }
        )

    strata = Counter(
        (
            str(row["metadata"]["source_dataset"]),
            int(row["metadata"]["ground_truth"]),
        )
        for row in prepared
    )
    if len(strata) != 4 or len(set(strata.values())) != 1:
        raise ValueError(f"selected rows are not balanced across four strata: {strata}")
    return prepared


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL atomically without changing row order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare(
    base_input: Path,
    output: Path,
    manifest_path: Path,
    *,
    total_rows: int,
) -> dict[str, Any]:
    """Materialize the paired ignored prompt file and its reconstruction manifest."""
    rows = prepare_paired_prompts(load_jsonl(base_input), total_rows=total_rows)
    write_jsonl(output, rows)
    prompt_set = load_prompt_set()
    candidate_template_sha256 = rows[0]["metadata"]["rendered_prompt_sha256"]
    manifest = {
        "canary_id": "kimi-k3-scalar-structured-pair-v1",
        "hypothesis": (
            "Strict scalar JSON decoding preserves Kimi K3 scores under the "
            "same scalar-only teacher interface, and that interface preserves "
            "the useful ranking of the original Prediction:0/1 contract."
        ),
        "selection": {
            "method": "first balanced interleaved rows from frozen teacher canary",
            "rows": total_rows,
            "strata": dict(
                sorted(
                    Counter(
                        f"{row['metadata']['source_dataset']}|"
                        f"{row['metadata']['ground_truth']}"
                        for row in rows
                    ).items()
                )
            ),
        },
        "base_input": {
            "path": str(base_input),
            "sha256": sha256_file(base_input),
        },
        "materialized_input": {
            "path": str(output),
            "sha256": sha256_file(output),
        },
        "prompt_set_id": prompt_set.prompt_set_id,
        "teacher_template_sha256": prompt_set.teacher.template_sha256,
        "candidate_first_prompt_sha256": candidate_template_sha256,
        "scalar_interface": {
            "path": str(
                SCALAR_INTERFACE_PATH.relative_to(Path(__file__).resolve().parents[2])
            ),
            "sha256": sha256_file(SCALAR_INTERFACE_PATH),
        },
        "conditions": {
            "unstructured": {
                "binary_output_mode": "scalar",
                "structured_output": False,
                "max_tokens": 1,
            },
            "structured": {
                "binary_output_mode": "scalar",
                "structured_output": True,
                "schema": {"type": "integer", "enum": [0, 1]},
                "max_tokens": 1,
            },
        },
        "acceptance": {
            "complete_pairs": total_rows,
            "minimum_score_pearson": 0.999,
            "minimum_score_spearman": 0.999,
            "maximum_absolute_score_delta": 0.01,
            "maximum_absolute_auroc_delta": 0.01,
            "maximum_threshold_flips": 0,
            "required_provider": "Makora",
            "minimum_reference_spearman": 0.95,
            "maximum_reference_auroc_degradation": 0.05,
        },
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _score_margin(row: dict[str, Any]) -> float:
    logprobs = row.get("label_logprobs") or {}
    return float(logprobs["1"]) - float(logprobs["0"])


def _generated_label(row: dict[str, Any]) -> int:
    text = str(row.get("text", "")).strip()
    if text in {"0", "1"}:
        return int(text)
    for label in ("0", "1"):
        if text.endswith(f"Prediction:{label}"):
            return int(label)
    else:
        raise ValueError(f"score row {row.get('id')!r} is not scalar: {text!r}")


def compare_paired_scores(
    input_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    unstructured_rows: list[dict[str, Any]],
    structured_rows: list[dict[str, Any]],
    *,
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    """Compare paired scalar scores and evaluate the frozen acceptance rule."""
    inputs = {str(row["id"]): row for row in input_rows}
    reference = {
        str(row["id"]): row
        for row in reference_rows
        if str(row["id"]) in inputs
    }
    unstructured = {str(row["id"]): row for row in unstructured_rows}
    structured = {str(row["id"]): row for row in structured_rows}
    if len(inputs) != len(input_rows):
        raise ValueError("input rows contain duplicate IDs")
    if (
        set(reference) != set(inputs)
        or set(unstructured) != set(inputs)
        or set(structured) != set(inputs)
    ):
        raise ValueError("all three score conditions must contain every input ID")

    joined = []
    for record_id, source in inputs.items():
        expected_hash = source["metadata"]["rendered_prompt_sha256"]
        base_hash = source["metadata"]["base_rendered_prompt_sha256"]
        baseline = reference[record_id]
        left = unstructured[record_id]
        right = structured[record_id]
        if baseline.get("prompt_sha256") != base_hash:
            raise ValueError(f"reference prompt mismatch for {record_id!r}")
        if left.get("prompt_sha256") != expected_hash:
            raise ValueError(f"unstructured prompt mismatch for {record_id!r}")
        if right.get("prompt_sha256") != expected_hash:
            raise ValueError(f"structured prompt mismatch for {record_id!r}")
        joined.append(
            {
                "id": record_id,
                "label": int(source["metadata"]["ground_truth"]),
                "source": str(source["metadata"]["source_dataset"]),
                "reference_score": float(baseline["score"]),
                "unstructured_score": float(left["score"]),
                "structured_score": float(right["score"]),
                "unstructured_margin": _score_margin(left),
                "structured_margin": _score_margin(right),
                "unstructured_label": _generated_label(left),
                "structured_label": _generated_label(right),
            }
        )
    frame = pd.DataFrame(joined)
    score_delta = frame["structured_score"] - frame["unstructured_score"]
    margin_delta = frame["structured_margin"] - frame["unstructured_margin"]
    threshold_flips = int(
        (
            (frame["structured_score"] >= 0.5)
            != (frame["unstructured_score"] >= 0.5)
        ).sum()
    )
    unstructured_auroc = float(
        roc_auc_score(frame["label"], frame["unstructured_score"])
    )
    structured_auroc = float(
        roc_auc_score(frame["label"], frame["structured_score"])
    )
    reference_auroc = float(roc_auc_score(frame["label"], frame["reference_score"]))
    score_pearson = float(
        np.corrcoef(frame["unstructured_score"], frame["structured_score"])[0, 1]
    )
    score_spearman = float(
        frame["unstructured_score"].corr(frame["structured_score"], method="spearman")
    )
    reference_spearman = float(
        frame["reference_score"].corr(frame["structured_score"], method="spearman")
    )
    unstructured_reference_spearman = float(
        frame["reference_score"].corr(
            frame["unstructured_score"], method="spearman"
        )
    )
    providers = {
        "unstructured": dict(
            sorted(
                Counter(
                    str(row.get("provider")) for row in unstructured_rows
                ).items()
            )
        ),
        "structured": dict(
            sorted(Counter(str(row.get("provider")) for row in structured_rows).items())
        ),
    }
    required_provider = str(acceptance["required_provider"])
    checks = {
        "complete_pairs": len(frame) == int(acceptance["complete_pairs"]),
        "score_pearson": score_pearson
        >= float(acceptance["minimum_score_pearson"]),
        "score_spearman": score_spearman
        >= float(acceptance["minimum_score_spearman"]),
        "absolute_score_delta": float(score_delta.abs().max())
        <= float(acceptance["maximum_absolute_score_delta"]),
        "absolute_auroc_delta": abs(structured_auroc - unstructured_auroc)
        <= float(acceptance["maximum_absolute_auroc_delta"]) + 1e-12,
        "threshold_flips": threshold_flips
        <= int(acceptance["maximum_threshold_flips"]),
        "provider": providers
        == {
            "unstructured": {required_provider: len(frame)},
            "structured": {required_provider: len(frame)},
        },
        "reference_spearman": reference_spearman
        >= float(acceptance["minimum_reference_spearman"]),
        "reference_auroc_degradation": reference_auroc - structured_auroc
        <= float(acceptance["maximum_reference_auroc_degradation"]),
    }
    by_source = []
    for source, subset in frame.groupby("source", sort=True):
        by_source.append(
            {
                "source": str(source),
                "rows": len(subset),
                "reference_auroc": float(
                    roc_auc_score(subset["label"], subset["reference_score"])
                ),
                "unstructured_auroc": float(
                    roc_auc_score(subset["label"], subset["unstructured_score"])
                ),
                "structured_auroc": float(
                    roc_auc_score(subset["label"], subset["structured_score"])
                ),
            }
        )
    return {
        "rows": len(frame),
        "accepted": all(checks.values()),
        "checks": checks,
        "acceptance": acceptance,
        "score_agreement": {
            "pearson": score_pearson,
            "spearman": score_spearman,
            "mean_absolute_delta": float(score_delta.abs().mean()),
            "maximum_absolute_delta": float(score_delta.abs().max()),
            "threshold_flips": threshold_flips,
            "generated_label_disagreements": int(
                (frame["structured_label"] != frame["unstructured_label"]).sum()
            ),
        },
        "margin_agreement": {
            "mean_absolute_delta": float(margin_delta.abs().mean()),
            "maximum_absolute_delta": float(margin_delta.abs().max()),
        },
        "auroc": {
            "reference": reference_auroc,
            "unstructured": unstructured_auroc,
            "structured": structured_auroc,
            "delta": structured_auroc - unstructured_auroc,
            "structured_minus_reference": structured_auroc - reference_auroc,
        },
        "reference_agreement": {
            "unstructured_spearman": unstructured_reference_spearman,
            "unstructured_pearson": float(
                np.corrcoef(
                    frame["reference_score"], frame["unstructured_score"]
                )[0, 1]
            ),
            "unstructured_mean_absolute_delta": float(
                (frame["unstructured_score"] - frame["reference_score"])
                .abs()
                .mean()
            ),
            "unstructured_maximum_absolute_delta": float(
                (frame["unstructured_score"] - frame["reference_score"])
                .abs()
                .max()
            ),
            "structured_spearman": reference_spearman,
            "structured_pearson": float(
                np.corrcoef(frame["reference_score"], frame["structured_score"])[0, 1]
            ),
            "structured_mean_absolute_delta": float(
                (frame["structured_score"] - frame["reference_score"]).abs().mean()
            ),
            "structured_maximum_absolute_delta": float(
                (frame["structured_score"] - frame["reference_score"]).abs().max()
            ),
        },
        "providers": providers,
        "by_source": by_source,
        "hard_accuracy": {
            "reference": float(
                ((frame["reference_score"] >= 0.5) == frame["label"]).mean()
            ),
            "unstructured": float(
                ((frame["unstructured_score"] >= 0.5) == frame["label"]).mean()
            ),
            "structured": float(
                ((frame["structured_score"] >= 0.5) == frame["label"]).mean()
            ),
        },
        "unique_scores": {
            "reference": int(frame["reference_score"].nunique()),
            "unstructured": int(frame["unstructured_score"].nunique()),
            "structured": int(frame["structured_score"].nunique()),
        },
        "reported_cost_usd": {
            "unstructured": float(
                sum(
                    float((row.get("usage") or {}).get("cost") or 0)
                    for row in unstructured_rows
                )
            ),
            "structured": float(
                sum(
                    float((row.get("usage") or {}).get("cost") or 0)
                    for row in structured_rows
                )
            ),
        },
        "pairs": joined,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--base-input", type=Path, default=DEFAULT_BASE_INPUT)
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_INPUT)
    prepare_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    prepare_parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    compare_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    compare_parser.add_argument(
        "--reference",
        type=Path,
        action="append",
        help="Original Prediction:0/1 score cache; repeat for recovery caches.",
    )
    compare_parser.add_argument(
        "--unstructured", type=Path, default=DEFAULT_UNSTRUCTURED
    )
    compare_parser.add_argument("--structured", type=Path, default=DEFAULT_STRUCTURED)
    compare_parser.add_argument("--output", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        manifest = prepare(
            args.base_input,
            args.output,
            args.manifest,
            total_rows=args.rows,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    summary = compare_paired_scores(
        load_jsonl(args.input),
        [
            row
            for path in (
                args.reference
                or [
                    Path("results/tool_trajectory_monitoring/kimi_k3_canary.jsonl"),
                    Path(
                        "results/tool_trajectory_monitoring/"
                        "kimi_k3_canary_retry64.jsonl"
                    ),
                ]
            )
            for row in load_jsonl(path)
        ],
        load_jsonl(args.unstructured),
        load_jsonl(args.structured),
        acceptance=manifest["acceptance"],
    )
    atomic_write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
