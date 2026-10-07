"""Compatibility alias for :mod:`gleipnir.evaluation.decision_surface`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("gleipnir.evaluation.decision_surface")
