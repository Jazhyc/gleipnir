"""Report another GPU model name to the serving stack (hardware trials only).

The optimized stack's kernel receipts are bound to the GPU model they were
validated on, and the workers compare that name with
``torch.cuda.get_device_name()``. To try the stack on another GPU without
editing hash-bound sources, ``run.py --gpu-alias NAME`` puts this directory
first on ``PYTHONPATH`` and sets ``GLEIPNIR_GPU_ALIAS``; the hook then makes
``torch.cuda.get_device_name`` return that name. Nothing else changes, so a
kernel that does not run on the real GPU still fails. Python imports only the
first ``sitecustomize`` on the path, so the hook then runs the next one (the
tokenizer hook).
"""

import importlib.abc
import importlib.util
import os
import runpy
import sys
from pathlib import Path

TARGET = "torch.cuda"
HERE = Path(__file__).resolve().parent


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self, alias: str) -> None:
        self.alias = alias

    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        execute = spec.loader.exec_module
        alias = self.alias

        def exec_module(module):
            execute(module)
            module.get_device_name = lambda device=None: alias

        spec.loader.exec_module = exec_module
        return spec


def chain() -> None:
    for entry in sys.path:
        candidate = Path(entry or ".").resolve() / "sitecustomize.py"
        if candidate.parent != HERE and candidate.is_file():
            runpy.run_path(str(candidate))
            return


if os.environ.get("GLEIPNIR_GPU_ALIAS"):
    sys.meta_path.insert(0, Finder(os.environ["GLEIPNIR_GPU_ALIAS"]))
chain()
