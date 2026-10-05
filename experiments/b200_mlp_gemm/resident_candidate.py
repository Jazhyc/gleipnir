"""Select native grouped Q/K heads for the resident GDN training screen."""

import importlib

import gleipnir.grouped_gdn as integration
from experiments.b200_mlp_gemm import grouped_candidate

# Each request archives these exact sources; reload compatible pilot fixes
# without replacing the resident model or retaining failed module globals.
importlib.reload(integration)
importlib.reload(grouped_candidate)
intervention = grouped_candidate.intervention
validate = grouped_candidate.validate

__all__ = ["intervention", "validate"]
