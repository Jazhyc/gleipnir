"""Compatibility alias for :mod:`gleipnir.evaluation.calibration`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("gleipnir.evaluation.calibration")
