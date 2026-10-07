"""Lazy aliases for moved modules still referenced by frozen runners."""

from __future__ import annotations

import sys
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import CodeType, ModuleType

MODULE_ALIASES = {
    "gleipnir.qwen35_loftq": "gleipnir.training.qwen35_loftq",
    "gleipnir.openrouter": "gleipnir.teachers.openrouter",
    "gleipnir.openrouter_cli": "gleipnir.teachers.openrouter_cli",
    "gleipnir.prefix_cache": "gleipnir.teachers.prefix_cache",
    "gleipnir.prefix_audit": "gleipnir.teachers.prefix_audit",
    "gleipnir.judge_injection": "gleipnir.data.judge_injection",
    "gleipnir.prefix_boundaries": "gleipnir.data.prefix_boundaries",
    "gleipnir.prefix_sampling": "gleipnir.data.prefix_sampling",
    "gleipnir.campaign_status": "gleipnir.campaigns.status",
    "gleipnir.binary_evaluation": "gleipnir.evaluation.binary",
    "gleipnir.metrics": "gleipnir.evaluation.metrics",
    "gleipnir.calibration": "gleipnir.evaluation.calibration",
    "gleipnir.decision_surface": "gleipnir.evaluation.decision_surface",
    "gleipnir.evaluation_lanes": "gleipnir.evaluation.lanes",
    "gleipnir.evaluation_shards": "gleipnir.evaluation.shards",
    "gleipnir.evaluation_watchdog": "gleipnir.evaluation.watchdog",
    "gleipnir.judge_injection_metrics": "gleipnir.evaluation.preferences",
    "gleipnir.monitoring_scoring": "gleipnir.evaluation.scoring",
    "gleipnir.evaluation.campaign": "gleipnir.monitoring_campaign_evaluation",
    "gleipnir.adaptive_microbatching": "gleipnir.training.adaptive_microbatching",
    "gleipnir.binary_task_training": "gleipnir.training.binary_tasks",
    "gleipnir.branch_model": "gleipnir.training.branch_model",
    "gleipnir.branch_trainer": "gleipnir.training.branch_trainer",
    "gleipnir.branch_training": "gleipnir.training.branches",
    "gleipnir.distributed_training": "gleipnir.training.distributed",
    "gleipnir.mil": "gleipnir.training.mil",
    "gleipnir.packed_training": "gleipnir.training.packed",
    "gleipnir.prefix_loss": "gleipnir.training.prefix_loss",
    "gleipnir.systems_artifacts": "gleipnir.training.artifacts",
    "gleipnir.training_execution_audit": "gleipnir.training.execution_audit",
    "gleipnir.branch_data": "gleipnir.data.branches",
    "gleipnir.monitoring_campaign_data": "gleipnir.data.monitoring",
    "gleipnir.monitoring_exclusions": "gleipnir.data.exclusions",
    "gleipnir.nested_subsets": "gleipnir.data.nested_subsets",
    "gleipnir.transcript_injection": "gleipnir.data.transcript_injection",
    "gleipnir.monitoring_campaign_training": "gleipnir.campaigns.training",
    "gleipnir.monitoring_systems_screen": "gleipnir.campaigns.systems_screen",
    "gleipnir.monitoring_training_command": "gleipnir.campaigns.training_command",
    "gleipnir.staged_lanes": "gleipnir.campaigns.lanes",
}

_CLI_MODULES = {"gleipnir.openrouter_cli", "gleipnir.monitoring_systems_screen"}


def canonical_source_reference(reference: str) -> str:
    """Translate a live relative source path from an archived source-list key.

    Use this when constructing new source snapshots, not when reading archived
    files or checking an old artifact's checksum. Unmoved references are exact.
    """
    for legacy, canonical in MODULE_ALIASES.items():
        if reference == "src/" + legacy.replace(".", "/") + ".py":
            return "src/" + canonical.replace(".", "/") + ".py"
    return reference


def _source_path(target: str) -> str:
    relative = target.removeprefix("gleipnir.").replace(".", "/") + ".py"
    return str(Path(__file__).parent / relative)


class _AliasLoader(Loader):
    def __init__(self, target: str) -> None:
        self.target = target

    def exec_module(self, module: ModuleType) -> None:
        """Reuse the canonical object without changing its import metadata."""
        sys.modules[module.__name__] = import_module(self.target)

    def get_code(self, fullname: str) -> CodeType:
        """Support historical command modules through ``python -m``."""
        source = (
            "from importlib import import_module\n"
            f"module = import_module({self.target!r})\n"
        )
        if fullname in _CLI_MODULES:
            source += "raise SystemExit(module.main())\n"
        return compile(source, _source_path(self.target), "exec")


class _AliasFinder(MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        destination = MODULE_ALIASES.get(fullname)
        if destination is None:
            return None
        return ModuleSpec(
            fullname,
            _AliasLoader(destination),
            origin=_source_path(destination),
        )


def install_aliases() -> None:
    """Register only these module names, without importing their dependencies."""
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
