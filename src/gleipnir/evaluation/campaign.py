"""Alias for the source-bound campaign evaluator retained at its recorded path."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("gleipnir.monitoring_campaign_evaluation")
