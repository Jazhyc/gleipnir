"""Report another GPU model name to the serving stack (hardware trials only).

The optimized stack's kernel receipts are bound to the GPU model they were
validated on, and the workers compare that name with
``torch.cuda.get_device_name()``. To try the stack on another GPU without
editing hash-bound sources, ``run.py --gpu-alias NAME`` puts this directory
first on ``PYTHONPATH`` and sets ``GLEIPNIR_GPU_ALIAS``; the hook then makes
``torch.cuda.get_device_name`` return that name. ``--gpu-capability 10.0``
also sets ``GLEIPNIR_GPU_CAPABILITY``, which ``get_device_capability`` then
returns to callers in the ``gleipnir`` package: the MXFP8 attention path
requires (10, 0) and a B300 reports (10, 3). Other callers get the real
capability, because Triton compiles for the capability it is told and an
``sm_100`` binary does not load on SM103. Nothing else changes, so a kernel
that does not run on the real GPU still fails. Python imports only the first
``sitecustomize`` on the path, so the hook then runs the next one (the
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
            capability = os.environ.get("GLEIPNIR_GPU_CAPABILITY")
            if capability:
                pair = tuple(int(part) for part in capability.split("."))
                real = module.get_device_capability

                def get_device_capability(device=None):
                    caller = sys._getframe(1).f_globals.get("__name__", "")
                    if caller.split(".")[0] == "gleipnir":
                        return pair
                    return real(device)

                module.get_device_capability = get_device_capability

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
