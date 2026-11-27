"""Independent experiment packages."""

# Register historical import names (``gleipnir._compat``) before any experiment
# module resolves, so moved modules keep their old ``experiments.*`` names in a
# cold process. Isolated runtimes without the package still import experiments.
try:
    import gleipnir  # noqa: F401
except ImportError:
    pass
