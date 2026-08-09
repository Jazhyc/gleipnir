#!/usr/bin/env python3
"""Compatibility wrapper for the installed ``gleipnir-openrouter`` command."""

from gleipnir.openrouter_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
