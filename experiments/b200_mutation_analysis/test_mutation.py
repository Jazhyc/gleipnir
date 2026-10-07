"""Runtime scalar/None positions must match TTIR and real mutations."""

import itertools
from types import SimpleNamespace

import pytest

from gleipnir.serving.triton_mutation import KERNEL, adapt_analysis_source, is_target

SOURCE = """
def generate_ttir(kernel, ordered_args):
    def get_tensor_names(name, arg):
        return [name] if isinstance(arg, Tensor) else []
    def is_tensor_like_arg(arg):
        return isinstance(arg, Tensor)
    ordered_tensor_names = list(itertools.chain.from_iterable(
        get_tensor_names(name, arg) for name, arg in ordered_args.items()
    ))
    constants = {
        name: arg for name, arg in ordered_args.items() if not is_tensor_like_arg(arg)
    }
    return ordered_tensor_names, constants
"""


class Tensor:
    pass


def test_runtime_scalars_keep_positional_slots_and_are_not_constants():
    namespace = {"itertools": itertools, "Tensor": Tensor}
    exec(adapt_analysis_source(SOURCE), namespace)
    kernel = SimpleNamespace(
        params=[
            SimpleNamespace(is_constexpr=v) for v in [False, False, False, True, False]
        ]
    )
    names, constants = namespace["generate_ttir"](
        kernel,
        {
            "input": Tensor(),
            "stride": 16,
            "output": Tensor(),
            "BLOCK": 256,
            "optional": None,
        },
    )
    assert names == ["input", "stride", "output"]
    assert constants == {"BLOCK": 256, "optional": None}


def test_changed_source_fails_closed():
    with pytest.raises(ValueError, match="source changed"):
        adapt_analysis_source(
            SOURCE.replace("if not is_tensor_like_arg(arg)", "if True")
        )


def test_scope_is_known_kernel_only_including_autotuners():
    kernel = SimpleNamespace(fn=SimpleNamespace(__name__=KERNEL))
    assert is_target(kernel) and is_target(SimpleNamespace(fn=kernel))
    assert not is_target(SimpleNamespace(fn=SimpleNamespace(__name__="other")))
