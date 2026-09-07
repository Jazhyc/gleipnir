"""Parallel GPU lanes with fail-closed barriers between experiment stages."""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor


def run_staged_lanes[T](
    stages: Sequence[Sequence[T]],
    worker: Callable[[int, T], None],
    on_stage_complete: Callable[[int], None],
) -> None:
    """Advance only after every lane and the stage-completion audit succeed."""
    if not stages or any(not stage for stage in stages):
        raise ValueError("stages and lanes must be nonempty")
    for index, stage in enumerate(stages):
        with ThreadPoolExecutor(max_workers=len(stage)) as pool:
            futures = [
                pool.submit(worker, gpu, entry) for gpu, entry in enumerate(stage)
            ]
            for future in futures:
                future.result()
        on_stage_complete(index)
