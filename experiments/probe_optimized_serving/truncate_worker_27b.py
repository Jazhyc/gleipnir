"""The Gleipnir 27B optimized worker, serving activation probes after block 31.

``QuantFullWorker`` with the 4B truncation (``truncate_worker.truncate``):
the decoder stops after block 31 of 64, the final norm returns the residual
stream, and ``GLEIPNIR_PROBE_BANK`` installs the probe pooler. Its last-query
attention and tail sit on block 63, which never runs.
"""

from __future__ import annotations

from experiments.probe_optimized_serving.truncate_worker import truncate
from gleipnir.serving.quant_stack import QuantFullWorker

# A constant, not an option: this file's hash joins the serving compile
# identity (start.py), so truncated and full-depth graphs never share an AOT
# compile-cache entry.
END_LAYER = 32


class TruncatedQuantFullWorker(QuantFullWorker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        truncate(self, 64, END_LAYER)
        return result
