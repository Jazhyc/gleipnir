"""Selected merged-weight QAT runtime with arithmetic-preserving optimizations."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from gleipnir.training import quantization_recovery
from gleipnir.training.qat_optimization import optimized_path


@contextmanager
def attention_runtime(model: Any) -> Iterator[None]:
    """Use serving MXFP8 forward, FA4 surrogate backward and lean MLP/GDN work.

    Install quantization_recovery projections first. Retain decoded FP8/FP32
    reference contractions; native FP8 GEMMs and native MXFP8 backward remain
    separate experimental variants.
    """
    if not hasattr(model, "_gleipnir_qad_groups"):
        raise ValueError("install merged QAT projections before entering the runtime")
    with (
        quantization_recovery.attention_runtime(),
        optimized_path(model, fp8=False, mlp=True, gdn=True),
    ):
        yield
