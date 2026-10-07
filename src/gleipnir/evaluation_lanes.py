"""Compatibility alias for :mod:`gleipnir.evaluation.lanes`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("gleipnir.evaluation.lanes")
