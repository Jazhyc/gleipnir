"""Frozen selection and paired numerical checks for the workstation campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("results/local_inference")
DATA = Path("data/local_inference")
CONFIG = Path("experiments/local_inference/config.json")


def parity_override(config: dict, passed: bool) -> str | None:
    """Allow an explicitly documented diagnostic, never relabel a failed gate."""
    reason = config.get("diagnostic_parity_override")
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("Diagnostic parity override requires a nonempty reason")
    if not passed and reason is None:
        raise RuntimeError("vLLM serving parity failed")
    return reason if not passed else None


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def read_rows(path: Path = DATA / "subset.jsonl") -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def select_subset(
    rows: list[dict[str, Any]], counts: dict[str, int], seed: int
) -> list[dict[str, Any]]:
    """Stratify into equal-count length bins with deterministic within-bin draws."""
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate input identities")
    selected = []
    for group, count in sorted(counts.items()):
        candidates = sorted(
            [r for r in rows if f"{r['source']}:{r['label']}" == group],
            key=lambda r: (r["tokens"], r["id"]),
        )
        if count < 4 or len(candidates) < count:
            raise ValueError(f"insufficient rows for {group}")
        for quartile in range(4):
            pool = candidates[
                len(candidates) * quartile // 4 : len(candidates) * (quartile + 1) // 4
            ]
            quota = count // 4 + (quartile < count % 4)
            if len(pool) < quota:
                raise ValueError("insufficient quartile population")
            pool.sort(key=lambda r: digest(f"{seed}:{r['id']}"))
            selected.extend(dict(r, length_quartile=quartile) for r in pool[:quota])
    return sorted(selected, key=lambda r: digest(f"order:{seed}:{r['id']}"))


def canary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Four shortest actual subset prompts, one per source/label group."""
    groups = sorted({(r["source"], r["label"]) for r in rows})
    return [
        min(
            (r for r in rows if (r["source"], r["label"]) == group),
            key=lambda r: (r["tokens"], r["id"]),
        )
        for group in groups
    ]


def compare(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    a, b = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not a.size:
        raise ValueError("score coverage mismatch")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite scores")
    error = np.abs(a - b)
    correlation = (
        float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
    )
    return {
        "mean_absolute_error": float(error.mean()),
        "max_absolute_error": float(error.max()),
        "p95_absolute_error": float(np.quantile(error, 0.95)),
        "correlation": correlation,
        "threshold_flips": int(((a >= 0.5) != (b >= 0.5)).sum()),
    }


def parity_passes(report: dict[str, Any], limits: dict[str, float]) -> bool:
    return (
        report["mean_absolute_error"] <= limits["mean_absolute_error"]
        and report["max_absolute_error"] <= limits["max_absolute_error"]
        and (
            report["correlation"] is None
            or report["correlation"] >= limits["min_correlation"]
        )
    )
