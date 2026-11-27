"""Reuse same-worker validation only after an identical candidate completed warm."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def reusable_validation(
    root: Path,
    *,
    integration_sha256: str,
    merged_inputs: bool,
    worker_pid: int,
    initial_master: str,
    physical_contract: list[dict],
) -> dict | None:
    """Source, arithmetic variant, masters and full batch contract define identity."""
    for path in sorted(root.glob("*/candidate_validation.json")):
        result_path = path.parent / "receipt.json"
        if not result_path.exists():
            continue
        validation = json.loads(path.read_text())
        result = json.loads(result_path.read_text())
        if (
            validation.get("accepted_for_timing") is True
            and validation.get("integration_source_sha256") == integration_sha256
            and validation.get("installation", {}).get("merged_qkv_z") is merged_inputs
            and validation.get("initial_master_sha256") == initial_master
            and validation.get("masters_unchanged") is True
            and validation.get("optimizer_updates") == 0
            and result.get("status") == "complete"
            and result.get("variant") == "candidate"
            and result.get("pid") == worker_pid
            and result.get("initial_master_sha256") == initial_master
            and result.get("physical_contract") == physical_contract
            and result.get("measured_updates_warm") is True
            and len(result.get("step_seconds", [])) == 20
        ):
            return {
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "validation": validation,
            }
    return None
