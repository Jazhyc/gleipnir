"""Compatibility alias for :mod:`gleipnir.evaluation.shards`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("gleipnir.evaluation.shards")
