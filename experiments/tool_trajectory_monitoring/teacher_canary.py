"""Prepare and summarize the bounded Kimi K3 teacher-logit canary."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from gleipnir.metrics import evaluate_binary_monitor

SOURCE_REPO = "Syghmon/blackboxmonitorsMATS"
SOURCE_REVISION = "ccb77e37589baaee1f0e16ac9ab89ceac8f6cc8a"
SOURCE_LICENSE = "research-use-only"
SOURCE_FILES = (
    "gemini_synthetic_1000.parquet",
    "unified_dataset_1000.parquet",
)
TOKENIZER_ID = "Qwen/Qwen3.5-9B"
TOKENIZER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
DEFAULT_SEED = 20260830
DEFAULT_MAX_INPUT_TOKENS = 28_000


def sha256_file(path: Path) -> str:
    """Hash one source or materialized artifact without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sample_key(seed: int, source: str, label: int, row_id: str) -> str:
    """Create a deterministic pseudo-random ordering within one stratum."""
    value = f"{seed}\0{source}\0{label}\0{row_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_balanced_candidates(
    candidates: list[dict[str, Any]],
    *,
    total_rows: int,
    seed: int,
    sources: Iterable[str],
) -> list[dict[str, Any]]:
    """Select equal source-label strata and interleave them for bounded canaries."""
    source_names = tuple(sorted(sources))
    stratum_count = len(source_names) * 2
    if total_rows < stratum_count or total_rows % stratum_count:
        raise ValueError(
            f"total_rows must be a positive multiple of {stratum_count}"
        )
    per_stratum = total_rows // stratum_count
    selected: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for source in source_names:
        for label in (0, 1):
            eligible = [
                row
                for row in candidates
                if row["source_dataset"] == source and row["ground_truth"] == label
            ]
            eligible.sort(
                key=lambda row: stable_sample_key(
                    seed,
                    source,
                    label,
                    str(row["id"]),
                )
            )
            if len(eligible) < per_stratum:
                raise ValueError(
                    f"stratum source={source!r} label={label} has "
                    f"{len(eligible)} rows; need {per_stratum}"
                )
            selected[(source, label)] = eligible[:per_stratum]

    interleaved = []
    for index in range(per_stratum):
        for source in source_names:
            for label in (0, 1):
                interleaved.append(selected[(source, label)][index])
    return interleaved


def stratified_bootstrap_auroc(
    frame: pd.DataFrame,
    *,
    resamples: int = 5_000,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Bootstrap AUROC while preserving every source-label stratum size."""
    if resamples < 1:
        raise ValueError("resamples must be positive")
    required = {"source", "label", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"bootstrap frame is missing columns: {sorted(missing)}")
    groups = [
        subset.index.to_numpy()
        for _, subset in frame.groupby(["source", "label"], sort=True)
    ]
    if len(frame["label"].unique()) != 2:
        raise ValueError("AUROC bootstrap requires both binary labels")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled_indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        sample = frame.loc[sampled_indices]
        estimates[index] = roc_auc_score(sample["label"], sample["score"])
    return {
        "low": float(np.quantile(estimates, 0.025)),
        "median": float(np.quantile(estimates, 0.5)),
        "high": float(np.quantile(estimates, 0.975)),
        "resamples": resamples,
        "seed": seed,
    }


def _numeric_sum(rows: list[dict[str, Any]], *keys: str) -> float:
    total = 0.0
    for row in rows:
        value: Any = row
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, int | float):
            total += float(value)
    return total


def summarize_scored_rows(
    input_rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    *,
    bootstrap_resamples: int = 5_000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Join a teacher cache to its frozen labels and report canary diagnostics."""
    inputs = {str(row["id"]): row for row in input_rows}
    if len(inputs) != len(input_rows):
        raise ValueError("input rows contain duplicate IDs")
    scores = {str(row["id"]): row for row in scored_rows}
    if len(scores) != len(scored_rows):
        raise ValueError("scored rows contain duplicate IDs")
    unknown = set(scores) - set(inputs)
    if unknown:
        raise ValueError(f"scored rows contain unknown IDs: {sorted(unknown)[:3]}")

    joined = []
    for record_id, scored in scores.items():
        source = inputs[record_id]
        metadata = source["metadata"]
        if scored.get("prompt_sha256") != metadata["rendered_prompt_sha256"]:
            raise ValueError(f"prompt hash mismatch for id={record_id!r}")
        generated_match = re.search(r"([01])\s*>?\s*$", str(scored["text"]))
        if generated_match is None:
            raise ValueError(f"could not parse generated label for id={record_id!r}")
        joined.append(
            {
                "id": record_id,
                "source": metadata["source_dataset"],
                "label": int(metadata["ground_truth"]),
                "score": float(scored["score"]),
                "generated_label": int(generated_match.group(1)),
            }
        )
    if not joined:
        raise ValueError("no scored rows were available")

    frame = pd.DataFrame(joined)
    overall_frame = frame.assign(group="overall")
    overall = evaluate_binary_monitor(overall_frame, group_column="group")
    overall_group = overall["groups"][0]
    overall_group["auroc_ci95"] = stratified_bootstrap_auroc(
        frame,
        resamples=bootstrap_resamples,
        seed=seed,
    )

    by_source = evaluate_binary_monitor(frame, group_column="source")
    for group in by_source["groups"]:
        subset = frame[frame["source"] == group["group"]]
        group["auroc_ci95"] = stratified_bootstrap_auroc(
            subset,
            resamples=bootstrap_resamples,
            seed=seed,
        )

    score_rows = list(scores.values())
    setting_groups = []
    for settings_sha256 in sorted(
        {str(row.get("request_settings_sha256")) for row in score_rows}
    ):
        subset = [
            row
            for row in score_rows
            if str(row.get("request_settings_sha256")) == settings_sha256
        ]
        settings = subset[0].get("request_settings")
        setting_groups.append(
            {
                "request_settings_sha256": settings_sha256,
                "rows": len(subset),
                "max_tokens": (
                    settings.get("max_tokens") if isinstance(settings, dict) else None
                ),
                "prompt_tokens": int(_numeric_sum(subset, "usage", "prompt_tokens")),
                "cached_tokens": int(
                    _numeric_sum(subset, "cache_usage", "cached_tokens")
                ),
                "reported_cost_usd": _numeric_sum(subset, "usage", "cost"),
            }
        )
    prompt_tokens = int(_numeric_sum(score_rows, "usage", "prompt_tokens"))
    cached_tokens = int(_numeric_sum(score_rows, "cache_usage", "cached_tokens"))
    result = {
        "requested_rows": len(input_rows),
        "scored_rows": len(joined),
        "missing_rows": len(input_rows) - len(joined),
        "label_counts": {
            str(key): int(value)
            for key, value in sorted(Counter(frame["label"]).items())
        },
        "source_counts": {
            str(key): int(value)
            for key, value in sorted(Counter(frame["source"]).items())
        },
        "overall": overall_group,
        "by_source": by_source["groups"],
        "mean_source_auroc": by_source["macro"]["auroc"],
        "mean_source_pauroc_at_20": by_source["macro"]["pauroc_at_20"],
        "generated_label_accuracy": float(
            (frame["label"] == frame["generated_label"]).mean()
        ),
        "generated_label_auroc": float(
            roc_auc_score(frame["label"], frame["generated_label"])
        ),
        "score_quantiles_by_label": {
            str(label): {
                str(quantile): float(value)
                for quantile, value in frame.loc[frame["label"] == label, "score"]
                .quantile([0.0, 0.25, 0.5, 0.75, 1.0])
                .items()
            }
            for label in (0, 1)
        },
        "providers": dict(
            sorted(Counter(str(row.get("provider")) for row in score_rows).items())
        ),
        "models": dict(
            sorted(Counter(str(row.get("model")) for row in score_rows).items())
        ),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": int(
                _numeric_sum(score_rows, "usage", "completion_tokens")
            ),
            "cached_tokens": cached_tokens,
            "cache_write_tokens": int(
                _numeric_sum(score_rows, "cache_usage", "cache_write_tokens")
            ),
            "cache_read_fraction": (
                cached_tokens / prompt_tokens if prompt_tokens else 0.0
            ),
            "reported_cost_usd": _numeric_sum(score_rows, "usage", "cost"),
        },
        "request_settings_sha256": sorted(
            {str(row.get("request_settings_sha256")) for row in score_rows}
        ),
        "request_setting_groups": setting_groups,
        "prompt_template_sha256": sorted(
            {
                str(inputs[record_id]["metadata"]["prompt_template_sha256"])
                for record_id in scores
            }
        ),
    }
    return result


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def load_source_candidates(source_directory: Path) -> list[dict[str, Any]]:
    """Load only the fields used by the frozen canary materializer."""
    candidates = []
    for filename in SOURCE_FILES:
        path = source_directory / filename
        table = pq.read_table(
            path,
            columns=["id", "ground_truth", "formatted_transcript", "source_dataset"],
        )
        for row_index, row in enumerate(table.to_pylist()):
            row["source_file"] = filename
            row["source_row_index"] = row_index
            candidates.append(row)
    return candidates
