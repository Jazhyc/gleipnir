"""Released Hub monitors, pinned to the revisions under test.

Adapter revisions are the Hub commits reviewed on 2026-10-20; base revisions
come from each release's training_summary.json. The Hub 4B is the project's
"cached released Gleipnir 4B" OOD control (source-macro pAUROC@20 0.782350);
the adversarial baseline reproduces it to 0.0003.
"""

from __future__ import annotations

from typing import NamedTuple


class Monitor(NamedTuple):
    """A Hub monitor. `adapter=None` loads `base` alone (merged or untrained)."""

    adapter: str | None
    adapter_revision: str | None
    base: str
    base_revision: str


MONITORS = {
    "4b": Monitor(
        "Jazhyc/Gleipnir-4B",
        "411bdb7cf28bf153d03820f242c7f0668762fa6d",
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    ),
    # v0.3 (2026-10-20): tool trajectories only, three-seed average, QAD,
    # rank 384. Adapter SHA256 0716efc1...; BF16 Mean-OOD pAUROC@20 0.803876.
    "4b-v03": Monitor(
        "Jazhyc/Gleipnir-4B",
        "e36aa2bb864639da79173eafd0460a45d21e6bcb",
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    ),
    "9b": Monitor(
        "Jazhyc/Gleipnir-9B",
        "bd86dda25f227119ea648b44d9df0e4576c2407a",
        "Qwen/Qwen3.5-9B",
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    ),
    "27b": Monitor(
        "Jazhyc/Gleipnir-27B",
        "e5c4a7a65f41e6059b8723f74dac15410896e94b",
        "Qwen/Qwen3.8-27B",
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    ),
    # Merged BF16 checkpoints (2026-10-20): the v0.3 4B adapter (identical in
    # v0.4) and the v0.1 27B adapter (identical in v0.2), merged in FP32 into
    # the bases above and cast once to BF16. Revisions are the `v0.4`/`v0.2` tags.
    "4b-v03-merged": Monitor(
        None,
        None,
        "Jazhyc/Gleipnir-4B-BF16",
        "1157856e4136add70907d4b689f2d73da58f08fb",
    ),
    "27b-merged": Monitor(
        None,
        None,
        "Jazhyc/Gleipnir-27B-BF16",
        "d3e995c137e5a2a8637e6ecdde26c6666864e71a",
    ),
    # The untrained 4B base, scored with the student prompt.
    "4b-base": Monitor(
        None,
        None,
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    ),
    # 4B trained with 40% injected views, labels unchanged (2026-10-04; rank
    # 128, one seed). Master adapter SHA256 3ecfd0d3...
    "4b-ia40": Monitor(
        "Jazhyc/Gleipnir-4B-ToolTrajectories-InjectionAugmented40",
        "26638fd37e615e74e09613a4f961c0f2343f5e64",
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    ),
}
