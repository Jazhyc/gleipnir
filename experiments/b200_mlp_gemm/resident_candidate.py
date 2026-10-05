"""Select the scoped NVIDIA BF16 convolution intervention for hot loading."""

import importlib

import gleipnir.nvidia_causal_conv1d as integration
from experiments.b200_mlp_gemm import convolution_candidate, convolution_reuse

# Each request archives these exact sources; reload compatible pilot fixes
# without replacing the resident model or retaining failed module globals.
importlib.reload(integration)
importlib.reload(convolution_reuse)
importlib.reload(convolution_candidate)
intervention = convolution_candidate.intervention
validate = convolution_candidate.validate

__all__ = ["intervention", "validate"]
