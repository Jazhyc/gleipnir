"""Reusable infrastructure for Gleipnir monitoring experiments."""

from ._compat import install_aliases as _install_aliases
from ._environment import expose_interpreter_scripts as _expose_interpreter_scripts

__version__ = "0.1.0"

_install_aliases()
_expose_interpreter_scripts()
