"""Explicit process-local aliases for documented CuTe register-tensor renames."""

from typing import Any


def install_aliases(cute: Any) -> dict[str, str]:
    """Bridge deprecated names without changing arithmetic or installed files."""
    applied = {}
    for old, new in (
        ("make_fragment", "make_rmem_tensor"),
        ("make_fragment_like", "make_rmem_tensor_like"),
    ):
        if hasattr(cute, old):
            continue
        if not hasattr(cute, new):
            raise ValueError(f"missing CuTe compatibility target: {new}")
        setattr(cute, old, getattr(cute, new))
        applied[old] = new
    return applied
