"""Combine existing graph policy and exact classifier audits cooperatively."""

from experiments.b200_attention_gdn_serving.prefill_graph_worker import (
    PrefillGraphNativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker,
)
from experiments.b200_monitor_score.worker import MonitorScoreAuditWorker


class GraphMonitorScoreAuditWorker(
    PrefillGraphNativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker,
    MonitorScoreAuditWorker,
):
    """Each load hook runs once, with the native backbone loaded once."""
