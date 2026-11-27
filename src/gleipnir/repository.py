"""Locate the checkout and experiment directories without fixed path depths.

Experiment code finds the repository root by walking up from its own file, so
it keeps working when experiment directories are grouped into categories.
Experiment directory names are unique, so ``experiment_dir`` resolves a name
whether it sits directly under ``experiments/`` or one category level below.
"""

from __future__ import annotations

from pathlib import Path


def is_repo_root(path: Path) -> bool:
    """Return whether ``path`` is a Gleipnir checkout root."""
    return (path / "pyproject.toml").is_file() and (path / "experiments").is_dir()


def repo_root(start: str | Path) -> Path:
    """Return the nearest checkout root at or above ``start``.

    Pass ``__file__`` from experiment, script and test code.
    """
    path = Path(start).resolve()
    for candidate in (path, *path.parents):
        if is_repo_root(candidate):
            return candidate
    raise RuntimeError(
        f"no Gleipnir checkout (pyproject.toml beside experiments/) above {path}"
    )


def experiment_dir(name: str, root: str | Path | None = None) -> Path:
    """Return the directory of experiment ``name`` within ``root``.

    ``root`` defaults to the checkout containing this package. The experiment
    may live at ``experiments/<name>`` or ``experiments/<category>/<name>``.
    """
    if not name or "/" in name or name.startswith((".", "_")):
        raise ValueError(f"invalid experiment name: {name!r}")
    base = (repo_root(__file__) if root is None else Path(root)) / "experiments"
    flat = base / name
    if flat.is_dir():
        return flat
    matches = sorted(p for p in base.glob(f"*/{name}") if p.is_dir())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"experiment {name!r} not found under {base}")
    raise RuntimeError(f"experiment name {name!r} is ambiguous: {matches}")
