"""Select the scoped CPU dispatch/conversion intervention for hot loading."""

import importlib

import gleipnir.training_hotpath as integration
from experiments.b200_mlp_gemm import hotpath_candidate

# Each request archives these exact sources; reload compatible pilot fixes
# without replacing the resident model or retaining failed module globals.
importlib.reload(integration)
importlib.reload(hotpath_candidate)
intervention = hotpath_candidate.intervention
validate = hotpath_candidate.validate

__all__ = ["intervention", "validate"]
