"""27B low-precision steps below the released ``QuantFullWorker``."""

from gleipnir.serving.bf16_stack import LastQueryMixin
from gleipnir.serving.quant_stack import (
    Mxfp8AttentionMixin,
    ProducerFusionMixin,
    QuantAuditWorker,
    TunedGemmMixin,
)


class QuantNoTrimWorker(
    TunedGemmMixin, ProducerFusionMixin, Mxfp8AttentionMixin, QuantAuditWorker
):
    """``QuantFullWorker`` minus ``FastGdnMixin`` and ``LastQueryMixin``."""


class QuantLastQueryWorker(
    TunedGemmMixin,
    ProducerFusionMixin,
    LastQueryMixin,
    Mxfp8AttentionMixin,
    QuantAuditWorker,
):
    """``QuantFullWorker`` minus ``FastGdnMixin``, in the same mixin order."""


def _prefix_workers() -> dict:
    """Prefix-cached twins of the 4B workers, for ``prefix.py``.

    Built lazily: importing the 4B workers reads 4B-only receipts, which the
    27B steps above must not depend on.
    """
    from experiments.b200_attention_precision.worker import AttentionPrecisionWorker
    from experiments.b200_gdn_fast.worker import FastGdnWorker
    from experiments.b200_prefix_replay.worker import PrefixCachingMixin
    from gleipnir.serving.bf16_worker import Bf16Worker

    return {
        "PrefixBf16Worker": type(
            "PrefixBf16Worker", (PrefixCachingMixin, Bf16Worker), {}
        ),
        "PrefixQuantWorker": type(
            "PrefixQuantWorker", (PrefixCachingMixin, AttentionPrecisionWorker), {}
        ),
        "PrefixFastWorker": type(
            "PrefixFastWorker", (PrefixCachingMixin, FastGdnWorker), {}
        ),
    }


def __getattr__(name: str):
    if name.startswith("Prefix"):
        workers = _prefix_workers()
        globals().update(workers)
        return workers[name]
    raise AttributeError(name)
