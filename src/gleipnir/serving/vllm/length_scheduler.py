"""Opt-in, source-bound admission scheduler for the pinned pooling scorer."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

from vllm.logger import init_logger
from vllm.v1.core.sched import request_queue, scheduler

from gleipnir.serving.length_admission import AdmissionPolicy, LengthAdmissionMixin

logger = init_logger(__name__)


class LengthAwareScheduler(LengthAdmissionMixin, scheduler.Scheduler):
    """Preserve synchronous causal pooling; change waiting admission only."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = json.loads(os.environ["GLEIPNIR_LENGTH_ADMISSION_CONFIG"])
        if (
            version("vllm") != "0.24.0"
            or self.vllm_config.model_config.runner_type != "pooling"
            or self.scheduler_config.async_scheduling
            or self.scheduler_config.policy != "fcfs"
            or not self.scheduler_config.enable_chunked_prefill
        ):
            raise ValueError(
                "length admission requires pinned synchronous FCFS pooling"
            )
        for module, name in (
            (scheduler, "scheduler_sha256"),
            (request_queue, "request_queue_sha256"),
        ):
            path = Path(inspect.getfile(module))
            if hashlib.sha256(path.read_bytes()).hexdigest() != config[name]:
                raise ValueError(f"length admission upstream source drift: {name}")
        source = Path(__file__)
        from gleipnir.serving import length_admission

        for path, name in (
            (source, "integration_sha256"),
            (Path(inspect.getfile(length_admission)), "policy_sha256"),
        ):
            if hashlib.sha256(path.read_bytes()).hexdigest() != config[name]:
                raise ValueError(f"length admission implementation drift: {name}")
        self.admission_policy = AdmissionPolicy.from_dict(config)
        self.admission_now = time.time()
        logger.info(
            "Length admission initialized: %s", json.dumps(config, sort_keys=True)
        )

    def schedule(self, throttle_prefills: bool = False) -> scheduler.SchedulerOutput:
        self.order_waiting(time.time())
        return super().schedule(throttle_prefills)
