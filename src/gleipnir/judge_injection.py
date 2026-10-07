"""Compatibility alias for :mod:`gleipnir.data.judge_injection`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("gleipnir.data.judge_injection")
