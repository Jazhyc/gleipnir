"""Identity checks for same-worker convolution preparation after audit failure."""

from __future__ import annotations


def can_reuse_preparation(
    validation: dict,
    failure: dict,
    sources: dict,
    *,
    initial_master: str,
    worker_pid: int,
    baseline_pid: int,
) -> bool:
    """Reuse arithmetic preparation, never a failed quality check or state reset."""

    def identity(values):
        return {k: v for k, v in values.items() if k not in {"candidate", "reuse"}}

    return (
        worker_pid == baseline_pid
        and validation.get("accepted_for_timing") is True
        and validation.get("quality_selection_eligible") is False
        and validation.get("initial_master_sha256") == initial_master
        and validation.get("masters_unchanged") is True
        and validation.get("optimizer_updates") == 0
        and len(validation.get("steps", [])) == 20
        and identity(validation.get("source_sha256", {})) == identity(sources)
        and failure.get("baseline_restored") is True
        and failure.get("initial_master_sha256") == initial_master
    )
