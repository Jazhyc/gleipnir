"""Fix the pinned pooling scheduler's generated-token reservation at the cap."""

import hashlib
import inspect
import json
import os
from importlib.metadata import version
from pathlib import Path

from vllm.v1.core.sched import scheduler


def correct_pooling_reservation(instance) -> None:
    """Pooling emits no generated tokens; leave all other scheduler logic intact."""
    if instance.vllm_config.model_config.runner_type != "pooling":
        raise ValueError("boundary correction requires pooling")
    instance.num_sampled_tokens_per_step = 0


class PoolingContextScheduler(scheduler.Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        binding = json.loads(os.environ["GLEIPNIR_POOLING_BOUNDARY_CONFIG"])
        if (
            version("vllm") != "0.24.0"
            or self.scheduler_config.async_scheduling
            or self.scheduler_config.policy != "fcfs"
        ):
            raise ValueError("boundary correction requires pinned synchronous FCFS")
        for path, key in (
            (Path(inspect.getfile(scheduler)), "scheduler_sha256"),
            (Path(__file__), "integration_sha256"),
        ):
            if hashlib.sha256(path.read_bytes()).hexdigest() != binding[key]:
                raise ValueError(f"pooling boundary source drift: {key}")
        correct_pooling_reservation(self)
