"""Compatibility alias for :mod:`gleipnir.teachers.prefix_audit`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("gleipnir.teachers.prefix_audit")
