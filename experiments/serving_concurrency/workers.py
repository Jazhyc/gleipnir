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
