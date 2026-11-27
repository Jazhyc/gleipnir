"""Shared fixtures for the crossed-text tests.

`study` is defined in test_pipeline_guards.py; re-exporting it here lets other
test modules request it by name without importing it, which would shadow the
fixture argument.
"""

from experiments.adversarial.adversarial_crossed_text.test_pipeline_guards import (  # noqa: F401
    study,
)
