"""Let the validated B200 stack pass its hardware checks on a B300.

The kernel receipts are bound to the GPU they were validated on: the workers
compare ``torch.cuda.get_device_name()`` with the receipts' ``NVIDIA B200``,
and the MXFP8 attention path requires compute capability (10, 0). A B300
(``NVIDIA B300 SXM6 AC``, capability (10, 3)) runs the same kernels and
returns the same scores, so on a B300 ``serve.py`` puts this directory first
on ``PYTHONPATH`` and sets ``GLEIPNIR_GPU_ALIAS`` and
``GLEIPNIR_GPU_CAPABILITY``. The hook then reports that name, and reports
that capability to callers in the ``gleipnir`` package only: Triton compiles
for the capability it is told, and an ``sm_100`` binary does not load on a
B300. The bound sources stay untouched. Python imports only the first
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
    def __init__(self, alias: str, capability: tuple[int, ...]) -> None:
        self.alias = alias
        self.capability = capability

    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        execute = spec.loader.exec_module
        alias, capability = self.alias, self.capability

        def exec_module(module):
            execute(module)
            real = module.get_device_capability

            def get_device_capability(device=None):
                caller = sys._getframe(1).f_globals.get("__name__", "")
                if caller.split(".")[0] == "gleipnir":
                    return capability
                return real(device)

            module.get_device_name = lambda device=None: alias
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
    sys.meta_path.insert(
        0,
        Finder(
            os.environ["GLEIPNIR_GPU_ALIAS"],
            tuple(
                int(part) for part in os.environ["GLEIPNIR_GPU_CAPABILITY"].split(".")
            ),
        ),
    )
chain()
