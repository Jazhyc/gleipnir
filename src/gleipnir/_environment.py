"""Process environment defaults applied when the package is imported."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def expose_interpreter_scripts() -> None:
    """Put the running interpreter's script directory first on ``PATH``.

    Runners are usually started as ``.venv*/bin/python`` without activating
    the environment, so console scripts installed beside the interpreter
    (``ninja`` for FlashInfer and Triton JIT builds) are missing from
    ``PATH``. Kernel builders, including those in spawned engine workers, look
    them up by name. The directory is taken unresolved, because resolving the
    ``python`` symlink lands in uv's interpreter cache instead of the venv.
    """
    scripts = str(Path(sys.executable).parent.absolute())
    entries = os.environ.get("PATH", "").split(os.pathsep)
    if entries[:1] != [scripts]:
        os.environ["PATH"] = os.pathsep.join(
            [scripts, *(entry for entry in entries if entry and entry != scripts)]
        )
