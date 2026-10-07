"""Compatibility alias for :mod:`gleipnir.data.prefix_sampling`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("gleipnir.data.prefix_sampling")
