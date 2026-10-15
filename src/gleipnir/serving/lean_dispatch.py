"""Call gleipnir custom-op implementations without torch.library wrapper layers.

`torch.library.custom_op` routes every call through a Python `backend_impl`, a
`torch._disable_dynamo` wrapper and an output-aliasing check before reaching the
user function. At inference these add host time on each of ~220 calls per step.
This registers each op's original function directly on the CUDA dispatch key,
which takes precedence over the custom op's CompositeExplicitAutograd kernel.
The function, its arguments and its kernels are unchanged. `set_enabled(False)`
routes through the original wrapper chain (including the aliasing check) for
paired in-server A/B tests.
"""

from __future__ import annotations

from typing import Any

_STATE: dict[str, Any] = {"enabled": True, "library": None, "ops": []}


def set_enabled(enabled: bool) -> dict:
    _STATE["enabled"] = bool(enabled)
    return audit()


def audit() -> dict:
    return {"enabled": _STATE["enabled"], "ops": list(_STATE["ops"])}


def install(namespace: str = "gleipnir") -> list[str]:
    import inspect

    import torch
    from torch._library import utils
    from torch._library.custom_ops import OPDEFS

    if _STATE["library"] is not None:
        raise ValueError("lean dispatch is already installed")
    library = torch.library.Library(namespace, "IMPL")
    installed = []
    for qualname, opdef in sorted(OPDEFS.items()):
        if not qualname.startswith(f"{namespace}::"):
            continue
        # Only plain functional ops with one device-agnostic implementation.
        if opdef._is_inplace or opdef._is_out or set(opdef._backend_fns) != {None}:
            continue
        name = qualname.split("::", 1)[1]
        function = opdef._init_fn
        wrapped = opdef._backend_fns[None]

        def kernel(*args, _fn=function, _wrapped=wrapped, _name=qualname, **kwargs):
            if _STATE["enabled"]:
                return _fn(*args, **kwargs)
            result = _wrapped(*args, **kwargs)
            utils._c_check_aliasing_constraint(
                _name, args, kwargs, result, lambda: inspect.getmodule(_fn)
            )
            return result

        library.impl(name, kernel, "CUDA")
        installed.append(qualname)
    _STATE["library"] = library
    _STATE["ops"] = installed
    return installed
