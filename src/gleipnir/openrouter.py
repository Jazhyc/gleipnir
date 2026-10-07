"""Compatibility alias for :mod:`gleipnir.teachers.openrouter`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("gleipnir.teachers.openrouter")
