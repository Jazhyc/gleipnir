"""27B low-precision step without last-query attention or the lean GDN path."""

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
