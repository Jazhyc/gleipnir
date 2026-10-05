"""Reuse completed hotpath checks for matching resident integration and installer."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


def installer_identity(source: str) -> str:
    """Ignore profiling switches; bind the actual context installer implementation."""
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "installed"
    )
    return ast.dump(function, include_attributes=False)


def reusable_validation(
    root: Path,
    *,
    integration_sha256: str,
    installer_source: str,
    normalize: bool,
    async_inputs: bool,
    prepared_metadata: bool,
    worker_pid: int,
    initial_master: str,
    physical_contract: list[dict],
) -> dict | None:
    """Require completed warm updates and unchanged precision/transfer controls."""
    for path in sorted(root.glob("*/candidate_validation.json")):
        receipt = path.parent / "receipt.json"
        source = path.parent / "executed_hotpath_candidate.py"
        if not receipt.exists() or not source.exists():
            continue
        validation = json.loads(path.read_text())
        result = json.loads(receipt.read_text())
        if (
            validation.get("accepted_for_timing") is True
            and validation.get("integration_source_sha256") == integration_sha256
            and validation.get("installation", {}).get(
                "normalization_input_copy_removed"
            ) is normalize
            and validation.get("installation", {}).get(
                "nonblocking_trainer_inputs"
            ) is async_inputs
            and validation.get("installation", {}).get(
                "cpu_prepared_packing"
            ) is prepared_metadata
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
            and installer_identity(source.read_text())
            == installer_identity(installer_source)
        ):
            return {
                **validation,
                "performed_this_trial": False,
                "reuse_reference": str(path.relative_to(root)),
                "reuse_reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "validation_wall_seconds": 0.0,
            }
    return None
