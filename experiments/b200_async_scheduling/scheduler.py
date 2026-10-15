"""Async-scheduling variant of the 0.31 zero-output-token pooling correction."""

import hashlib
import inspect
import json
import os
from importlib.metadata import version
from pathlib import Path

from vllm.v1.core.sched import async_scheduler, scheduler


class Vllm031AsyncPoolingScheduler(async_scheduler.AsyncScheduler):
    """Upstream AsyncScheduler; pooling requests reserve no sampled token.

    The synchronous correction (`experiments/b200_vllm031/scheduler.py`) sets
    `num_sampled_tokens_per_step = 0` so a max-length prompt is not cut by one
    token. In AsyncScheduler the same value also sizes output placeholders,
    which pooling requests never fill, so zero keeps them balanced.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        binding = json.loads(os.environ["GLEIPNIR_VLLM031_SCHEDULER_BINDING"])
        if (
            version("vllm") != "0.31.0"
            or self.vllm_config.model_config.runner_type != "pooling"
            or not self.scheduler_config.async_scheduling
            or self.scheduler_config.policy != "fcfs"
        ):
            raise ValueError("async correction requires async FCFS pooling on 0.31")
        upstream = hashlib.sha256(
            Path(inspect.getfile(scheduler)).read_bytes()
        ).hexdigest()
        if upstream != binding["upstream_sha256"]:
            raise ValueError("0.31 pooling scheduler source drift: upstream_sha256")
        self.num_sampled_tokens_per_step = 0
