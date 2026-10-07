"""Compatibility alias for :mod:`gleipnir.campaigns.status`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("gleipnir.campaigns.status")
