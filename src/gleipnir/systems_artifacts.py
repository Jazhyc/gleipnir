"""Bounded, mutable weight storage for short systems-training diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def systems_scratch(root: Path | None = None) -> Path:
    """Return the shared scratch location, separate from immutable run receipts."""
    repository = root or Path(__file__).resolve().parents[2]
    return repository / "results/systems_training_scratch"


def save_systems_master(tensors: dict[str, Any], root: Path | None = None) -> dict:
    """Atomically replace the latest diagnostic master, preserving it on failure."""
    import torch

    directory = systems_scratch(root)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "fp32_master.pt"
    temporary = directory / f"fp32_master.{os.getpid()}.tmp"
    try:
        torch.save(tensors, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(target), "mutable": True, "retention": "latest_only"}
