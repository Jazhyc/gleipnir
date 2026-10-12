"""Join frozen binary teacher caches without conflating teacher/student prompts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from gleipnir.data.monitoring import file_hash, read_rows


def load_binary_targets(
    workload: list[dict[str, Any]],
    cache: Path,
    *,
    expected_sha256: str,
    teacher_model: str,
    prompt_contract: Literal["same", "privileged"],
) -> tuple[list[float], dict[str, Any]]:
    """Validate complete source membership and retain separate prompt provenance."""
    if prompt_contract not in {"same", "privileged"}:
        raise ValueError("unknown teacher prompt contract")
    if file_hash(cache) != expected_sha256:
        raise ValueError("teacher cache checksum mismatch")
    teachers = read_rows(cache)

    def key(row: dict[str, Any], field: str) -> tuple[str, str]:
        return row["dataset"], row[field]

    indexed = {key(row, "index"): row for row in teachers}
    workload_keys = [key(row, "id") for row in workload]
    if len(indexed) != len(teachers) or len(set(workload_keys)) != len(workload):
        raise ValueError("duplicate source identity")
    if set(indexed) != set(workload_keys):
        raise ValueError("teacher/workload population mismatch")
    targets, bindings = [], []
    for row, identity in zip(workload, workload_keys, strict=True):
        teacher = indexed[identity]
        if (
            teacher["teacher_model"] != teacher_model
            or teacher["label"] != row["label"]
        ):
            raise ValueError("teacher model/source label mismatch")
        teacher_hash = teacher["rendered_prompt_sha256"]
        student_hash = row["prompt_sha256"]
        for value in (teacher_hash, student_hash):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid prompt identity")
        if prompt_contract == "same" and teacher_hash != student_hash:
            raise ValueError("same-prompt teacher identity mismatch")
        p = float(teacher["soft_target"])
        probs = teacher["target_probs"]
        if (
            not math.isfinite(p)
            or not 0 <= p <= 1
            or not math.isclose(float(probs["positive"]), p, abs_tol=1e-7)
            or not math.isclose(float(probs["negative"]), 1 - p, abs_tol=1e-7)
        ):
            raise ValueError("invalid binary teacher distribution")
        targets.append(p)
        bindings.append(
            {
                "dataset": identity[0],
                "id": identity[1],
                "student_prompt_sha256": student_hash,
                "teacher_prompt_sha256": teacher_hash,
            }
        )
    return targets, {
        "cache_sha256": expected_sha256,
        "teacher_model": teacher_model,
        "prompt_contract": prompt_contract,
        "rows": len(targets),
        "bindings": bindings,
        "hard_labels_used_in_loss": False,
    }
