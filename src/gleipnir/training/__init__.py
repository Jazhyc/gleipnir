"""Training utilities with lazy compatibility for the former training module."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .optimizers import (
        MUON_LR_ADJUSTMENTS,
        MuonAdamW,
        configure_mean_loss_accumulation,
        muon_adamw_param_groups,
        muon_update_scale,
        zeropower_via_newtonschulz5,
    )

__all__ = [
    "MUON_LR_ADJUSTMENTS",
    "MuonAdamW",
    "configure_mean_loss_accumulation",
    "muon_adamw_param_groups",
    "muon_update_scale",
    "zeropower_via_newtonschulz5",
]


def __getattr__(name: str) -> Any:
    """Keep historical training imports without loading Torch for the package."""
    if name in __all__:
        return getattr(import_module(".optimizers", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
