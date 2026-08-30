"""Export a prompt-free public table from provider-separated teacher caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache_rows(paths: list[Path]) -> list[dict[str, Any]]:
    """Load disjoint cache shards and reject duplicate record IDs."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                record_id = str(row["id"])
                if record_id in seen:
                    raise ValueError(
                        f"duplicate id {record_id!r} at {path}:{line_number}"
                    )
                seen.add(record_id)
                rows.append(row)
    return rows


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    """Select join keys and teacher targets without publishing source inputs."""
    metadata = row["metadata"]
    label_logprobs = row["label_logprobs"]
    target_probs = row["target_probs"]
    usage = row["usage"]
    cache_usage = row["cache_usage"]
    request_settings = row["request_settings"]
    score = float(row["score"])
    positive_probability = float(target_probs["positive"])
    if abs(score - positive_probability) > 1e-12:
        raise ValueError(f"score/probability mismatch for id={row['id']!r}")
    if abs(float(target_probs["negative"]) + positive_probability - 1.0) > 1e-9:
        raise ValueError(f"target probabilities do not sum to one for id={row['id']!r}")
    text = str(row["text"]).strip()
    prediction = re.search(r"(?i)Prediction\s*:\s*<?([01])>?\s*$", text)
    if prediction is None:
        raise ValueError(f"invalid terminal prediction for id={row['id']!r}")
    return {
        "id": str(row["id"]),
        "source_dataset": str(metadata["source_dataset"]),
        "source_repo": str(metadata["source_repo"]),
        "source_revision": str(metadata["source_revision"]),
        "source_file": str(metadata["source_file"]),
        "source_row_index": int(metadata["source_row_index"]),
        "prompt_id": int(metadata["prompt_id"]),
        "original_prompt_id": int(metadata["original_prompt_id"]),
        "sample_index": int(metadata["sample_index"]),
        "raw_source": str(metadata["raw_source"]),
        "trajectory_sha256": str(metadata["trajectory_sha256"]),
        "rendered_prompt_sha256": str(metadata["rendered_prompt_sha256"]),
        "prompt_template_sha256": str(metadata["prompt_template_sha256"]),
        "model": str(row["model"]),
        "provider": str(row["provider"]),
        "request_settings_sha256": str(row["request_settings_sha256"]),
        "generated_label": int(prediction.group(1)),
        "score": score,
        "logprob_0": float(label_logprobs["0"]),
        "logprob_1": float(label_logprobs["1"]),
        "probability_0": float(target_probs["negative"]),
        "probability_1": positive_probability,
        "top_logprobs_json": json.dumps(row["top_logprobs"], sort_keys=True),
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": int(usage["completion_tokens"]),
        "cached_tokens": int(cache_usage["cached_tokens"]),
        "max_tokens": int(request_settings["max_tokens"]),
    }


def export_public_cache(
    score_paths: list[Path],
    summary_path: Path,
    card_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Write the sanitized Parquet table, metrics, card, and manifest."""
    rows = load_cache_rows(score_paths)
    public_rows = sorted((public_row(row) for row in rows), key=lambda row: row["id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "train.parquet"
    metrics_path = output_dir / "metrics.json"
    readme_path = output_dir / "README.md"
    pq.write_table(
        pa.Table.from_pylist(public_rows),
        table_path,
        compression="zstd",
        version="2.6",
    )
    shutil.copyfile(summary_path, metrics_path)
    shutil.copyfile(card_path, readme_path)
    manifest = {
        "schema_version": "gleipnir-kimi-logits-public-v1",
        "rows": len(public_rows),
        "contains_prompts_or_trajectories": False,
        "contains_ground_truth_labels": False,
        "providers": dict(
            sorted(Counter(row["provider"] for row in public_rows).items())
        ),
        "models": dict(sorted(Counter(row["model"] for row in public_rows).items())),
        "request_settings_sha256": sorted(
            {row["request_settings_sha256"] for row in public_rows}
        ),
        "prompt_template_sha256": sorted(
            {row["prompt_template_sha256"] for row in public_rows}
        ),
        "source_cache_files": [
            {"name": path.name, "sha256": sha256_file(path)} for path in score_paths
        ],
        "files": {
            "train.parquet": sha256_file(table_path),
            "metrics.json": sha256_file(metrics_path),
            "README.md": sha256_file(readme_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, action="append", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = export_public_cache(
        args.scores,
        args.summary,
        args.card,
        args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
