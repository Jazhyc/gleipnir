"""Single-token binary decision surfaces independent of label semantics."""

from collections.abc import Sequence
from typing import Any


def decision_token_ids(tokenizer: Any, tokens: Sequence[str] = ("0", "1")) -> list[int]:
    """Validate two distinct literal tokens, ordered as class zero and class one."""
    if (
        isinstance(tokens, str)
        or len(tokens) != 2
        or any(not isinstance(value, str) or not value for value in tokens)
    ):
        raise ValueError("binary targets require two nonempty strings")
    ids = []
    for value in tokens:
        encoded = tokenizer.encode(value, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"binary target {value!r} tokenized as {encoded}, expected one token"
            )
        ids.append(int(encoded[0]))
    if len(set(ids)) != 2:
        raise ValueError(f"binary targets must have distinct token ids, got {ids}")
    return ids
