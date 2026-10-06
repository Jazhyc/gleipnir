"""Retain the selected FP4 projection audits when replacing attention/GDN."""

from experiments.b200_attention_gdn_serving.fa4_worker import Fa4ServingAuditWorker
from experiments.b200_attention_gdn_serving.gdn_worker import GdnServingAuditWorker
from experiments.b200_attention_gdn_serving.mixed_worker import MixedServingAuditWorker


class MixedFa4ServingAuditWorker(MixedServingAuditWorker, Fa4ServingAuditWorker):
    """Use mixed projection loading with the native paged FA4 attention audit."""


class MixedGdnServingAuditWorker(GdnServingAuditWorker, MixedServingAuditWorker):
    """Select a validated recurrence backend around audited FP4 projections."""
