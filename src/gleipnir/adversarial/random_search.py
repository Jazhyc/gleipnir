"""Deterministic score-only suffix search for the APPS research evaluation."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any

from gleipnir.adversarial.python_code import check_comment_only_edit


def stable_seed(*parts: Any) -> int:
    return int.from_bytes(
        hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8]
    )


def append_comment(code: str, payload: str) -> str:
    """Append a removable full-line comment; preserve all original bytes."""
    if not payload.strip() or any(not 32 <= ord(c) <= 126 for c in payload):
        raise ValueError("payload must be nonempty printable ASCII")
    addition = ("" if code.endswith("\n") else "\n") + "# " + payload + "\n"
    edited = code + addition
    assert edited[: -len(addition)] == code
    # tokenize represents a missing final newline differently; canonicalize only
    # that sentinel for the semantic check, never the stored original program.
    check = check_comment_only_edit(code.rstrip("\n") + "\n", edited)
    if not check.ok:
        raise ValueError(check.reason)
    return edited


@dataclass(frozen=True)
class Candidate:
    pieces: tuple[int, ...]
    text: str
    tokens: int


class SuffixSpace:
    """Printable, round-tripping tokenizer pieces; no model or score access."""

    def __init__(self, tokenizer: Any, max_tokens: int = 20):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        special = set(tokenizer.all_special_ids)
        self.pool = []
        self.texts = {}
        for token_id in sorted(set(tokenizer.get_vocab().values()) - special):
            text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            if (
                text.strip()
                and len(text) <= 16
                and all(32 <= ord(c) <= 126 for c in text)
                and tokenizer.encode(text, add_special_tokens=False) == [token_id]
            ):
                self.pool.append(token_id)
                self.texts[token_id] = text
        if len(self.pool) < 2:
            raise ValueError("insufficient printable token vocabulary")

    def candidate(self, pieces: tuple[int, ...]) -> Candidate | None:
        text = "".join(self.texts[p] for p in pieces)
        count = len(self.tokenizer.encode(text, add_special_tokens=False))
        if 0 < count <= self.max_tokens:
            return Candidate(pieces, text, count)
        return None

    def propose(
        self, rng: random.Random, incumbent: Candidate | None, mutations: int = 1
    ) -> Candidate:
        for _ in range(1000):
            if incumbent is None:
                pieces = [rng.choice(self.pool) for _ in range(self.max_tokens)]
            else:
                pieces = list(incumbent.pieces)
                for position in rng.sample(range(len(pieces)), mutations):
                    replacement = rng.choice(self.pool)
                    while replacement == pieces[position]:
                        replacement = rng.choice(self.pool)
                    pieces[position] = replacement
            candidate = self.candidate(tuple(pieces))
            if candidate is not None and (
                incumbent is None or candidate.text != incumbent.text
            ):
                return candidate
        raise ValueError("could not construct a valid token-bounded suffix")


@dataclass
class SearchState:
    """Baseline is eligible as the best outcome, including no edit."""

    score: float
    best: Candidate | None = None
    queries: int = 0

    def observe(self, candidate: Candidate, score: float) -> bool:
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("invalid score")
        self.queries += 1
        improved = score < self.score
        if improved:
            self.score, self.best = score, candidate
        return improved


def proposal(
    space: SuffixSpace, state: SearchState, task: str, seed: int, arm: str, query: int
) -> Candidate:
    if arm not in ("adaptive", "independent") or query != state.queries + 1:
        raise ValueError("invalid arm or noncontiguous query")
    # Shared initial draw, scored separately and charged once to each arm.
    stream = "initial" if query == 1 else arm
    rng = random.Random(stable_seed("a3-v1", task, seed, stream, query))
    # Every eighth draw restarts globally to avoid permanent local trapping.
    local = arm == "adaptive" and query % 8 != 0 and state.best is not None
    mutations = 4 if query <= 16 else 2 if query <= 32 else 1
    return space.propose(rng, state.best if local else None, mutations)
