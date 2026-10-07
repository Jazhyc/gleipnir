"""Compatibility alias for :mod:`gleipnir.teachers.openrouter_cli`."""

import sys
from importlib import import_module

if __name__ == "__main__":
    raise SystemExit(import_module("gleipnir.teachers.openrouter_cli").main())

sys.modules[__name__] = import_module("gleipnir.teachers.openrouter_cli")
