"""The probe pooler's lookup key for a request (stdlib only)."""

from __future__ import annotations

import hashlib


def token_key(ids) -> str:
    """The keep-table key of a token-id sequence."""
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
