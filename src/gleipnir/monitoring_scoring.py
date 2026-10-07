"""Compatibility alias for :mod:`gleipnir.evaluation.scoring`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("gleipnir.evaluation.scoring")
