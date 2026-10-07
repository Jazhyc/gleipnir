"""Compatibility alias for :mod:`gleipnir.training.qwen35_loftq`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("gleipnir.training.qwen35_loftq")
