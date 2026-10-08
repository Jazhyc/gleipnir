"""Preserve the zero-output-token correction on the examined 0.31 scheduler."""

import hashlib
import inspect
import json
import os
from importlib.metadata import version
from pathlib import Path

from vllm.v1.core.sched import scheduler


class Vllm031PoolingScheduler(scheduler.Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        binding = json.loads(os.environ["GLEIPNIR_VLLM031_SCHEDULER_BINDING"])
        if (
            version("vllm") != "0.31.0"
            or self.vllm_config.model_config.runner_type != "pooling"
            or self.scheduler_config.async_scheduling
            or self.scheduler_config.policy != "fcfs"
        ):
            raise ValueError("0.31 correction requires synchronous FCFS pooling")
        for path, key in (
            (Path(inspect.getfile(scheduler)), "upstream_sha256"),
            (Path(__file__), "integration_sha256"),
        ):
            if hashlib.sha256(path.read_bytes()).hexdigest() != binding[key]:
                raise ValueError(f"0.31 pooling scheduler source drift: {key}")
        self.num_sampled_tokens_per_step = 0
