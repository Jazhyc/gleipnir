"""Isolate a compiler backward-autocast assumption in a canary subprocess."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def run_training_canary(
    entrypoint: Path, overrides: list[str], backward_autocast: str
) -> None:
    """Run the existing training entrypoint with a scoped compiler assumption."""
    if backward_autocast not in {"off", "same_as_forward"}:
        raise ValueError("unknown backward-autocast assumption")
    import torch._functorch.config as compiler_config

    original_argv = sys.argv
    try:
        sys.argv = [str(entrypoint), *overrides]
        with compiler_config.patch(backward_pass_autocast=backward_autocast):
            print(f"diagnostic_backward_pass_autocast={backward_autocast}", flush=True)
            runpy.run_path(str(entrypoint), run_name="__main__")
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    run_training_canary(Path(sys.argv[2]), sys.argv[3:], sys.argv[1])
