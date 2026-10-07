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
}


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
        """Support the historical annotation command through ``python -m``."""
        source = (
            "from importlib import import_module\n"
            f"module = import_module({self.target!r})\n"
        )
        if fullname == "gleipnir.openrouter_cli":
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
