"""Reproducible, label-blind sampling of one intermediate target per parent."""

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from gleipnir.prefix_cache import binary_score


def sample_parent_prefixes(
    candidates: Mapping[str, Sequence[str]], *, seed: int, epoch: int
) -> dict[str, str]:
    """Sample uniformly within parents, independent of input ordering.

    Parents without candidates have no sampled prefix. This only selects the
    auxiliary target: callers must retain every parent's full Kimi target.
    Repeated IDs are rejected because they would silently bias sampling.
    """
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    selected = {}
    seen = set()
    for parent in sorted(candidates):
        choices = sorted(candidates[parent])
        if len(set(choices)) != len(choices) or seen.intersection(choices):
            raise ValueError("prefix IDs must be globally unique")
        seen.update(choices)
        if not choices:
            continue
        identity = f"prefix-sampling-v1\0{seed}\0{epoch}\0{parent}".encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(identity).digest(), "big"))
        selected[parent] = choices[rng.randrange(len(choices))]
    return selected


def attach_sampled_prefix(
    parent: Mapping[str, Any], cached_prefix: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Attach auxiliary evidence without changing any full-trajectory field.

    The cache must first pass whole-cache provenance/completeness validation.
    Here we additionally verify the exact parent text and selected boundary.
    """
    result = dict(parent)
    if cached_prefix is None:
        return result
    if cached_prefix["parent_prompt_id"] != parent["prompt_id"]:
        raise ValueError("prefix parent mismatch")
    before, separator, remainder = parent["student_prompt"].partition(
        "<agent_trajectory>\n"
    )
    trajectory, closing, after = remainder.rpartition("</agent_trajectory>")
    if not separator or not closing:
        raise ValueError("missing student trajectory envelope")
    endpoint = cached_prefix["end_character"]
    if not isinstance(endpoint, int) or not 0 < endpoint < len(trajectory):
        raise ValueError("prefix must be a proper nonempty trajectory prefix")
    prefix = trajectory[:endpoint]
    if hashlib.sha256(prefix.encode()).hexdigest() != cached_prefix["prefix_sha256"]:
        raise ValueError("prefix text hash mismatch")
    target = binary_score(cached_prefix["logprob_0"], cached_prefix["logprob_1"])
    if not math.isclose(target, cached_prefix["score"], abs_tol=1e-12, rel_tol=0):
        raise ValueError("prefix target disagrees with raw logits")
    result.update(
        prefix_student_prompt=(
            before
            + separator
            + prefix
            + ("" if prefix.endswith("\n") else "\n")
            + closing
            + after
        ),
        prefix_soft_target=target,
        prefix_id=cached_prefix["id"],
        prefix_teacher_contract_sha256=cached_prefix["contract_sha256"],
    )
    return result
