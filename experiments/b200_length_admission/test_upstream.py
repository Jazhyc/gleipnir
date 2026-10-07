"""Exercise the installed vLLM scheduler and CPU KV allocator without a worker."""

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.config import SchedulerConfig
from vllm.pooling_params import PoolingParams
from vllm.v1.core.sched import request_queue, scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

from gleipnir.serving import length_admission
from gleipnir.serving.vllm import length_scheduler


def make_scheduler(mode, monkeypatch):
    config = {
        "mode": mode,
        "max_wait_seconds": 0.25,
        "length_buckets": [1024, 4096, 16384, 32768],
    }
    for module, key in (
        (scheduler, "scheduler_sha256"),
        (request_queue, "request_queue_sha256"),
        (length_admission, "policy_sha256"),
        (length_scheduler, "integration_sha256"),
    ):
        config[key] = hashlib.sha256(
            Path(inspect.getfile(module)).read_bytes()
        ).hexdigest()
    vllm_config = SimpleNamespace(
        scheduler_config=SchedulerConfig.default_factory(
            max_model_len=32768,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
            enable_chunked_prefill=True,
            async_scheduling=False,
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks=4096, enable_prefix_caching=False, mamba_cache_mode="none"
        ),
        model_config=SimpleNamespace(
            is_encoder_decoder=False,
            is_diffusion=False,
            max_model_len=32768,
            enable_return_routed_experts=False,
            runner_type="pooling",
        ),
        parallel_config=SimpleNamespace(
            data_parallel_index=0,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        observability_config=SimpleNamespace(kv_cache_metrics=False),
        additional_config={},
        lora_config=None,
        kv_events_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
        speculative_config=None,
        num_speculative_tokens=0,
        use_v2_model_runner=False,
    )
    kv = KVCacheConfig(
        num_blocks=4096,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layer"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=64,
                    dtype=torch.bfloat16,
                ),
            )
        ],
    )
    monkeypatch.setattr(length_scheduler.time, "time", lambda: 100.1)
    monkeypatch.setenv("GLEIPNIR_LENGTH_ADMISSION_CONFIG", json.dumps(config))
    return length_scheduler.LengthAwareScheduler(
        vllm_config,
        kv,
        SimpleNamespace(),
        16,
        mm_registry=SimpleNamespace(supports_multimodal_inputs=lambda _: False),
    )


def add(sched, name, length, arrival=100.0):
    req = Request(
        name, [1] * length, None, PoolingParams(task="classify"), arrival_time=arrival
    )
    sched.add_request(req)
    return req


@pytest.mark.parametrize("mode,first", [("length_aware", "short"), ("fcfs", "long")])
def test_real_upstream_admission_keeps_chunk_budget(mode, first, monkeypatch):
    sched = make_scheduler(mode, monkeypatch)
    add(sched, "long", 20000)
    add(sched, "short", 768)
    output = sched.schedule(False)
    assert output.scheduled_new_reqs[0].req_id == first
    assert sum(output.num_scheduled_tokens.values()) == 1024
    if mode == "length_aware":
        assert output.num_scheduled_tokens == {"short": 768, "long": 256}
    else:
        assert output.num_scheduled_tokens == {"long": 1024}


def test_real_upstream_age_promotion_and_cancellation(monkeypatch):
    sched = make_scheduler("length_aware", monkeypatch)
    add(sched, "old_long", 20000, arrival=99.0)
    add(sched, "short", 768)
    cancelled = add(sched, "cancelled", 200)
    sched.finish_requests(
        cancelled.request_id, scheduler.RequestStatus.FINISHED_ABORTED
    )
    output = sched.schedule()
    assert [r.req_id for r in output.scheduled_new_reqs] == ["old_long"]
    assert output.num_scheduled_tokens == {"old_long": 1024}


def test_real_engine_prefill_throttle_is_forwarded(monkeypatch):
    sched = make_scheduler("length_aware", monkeypatch)
    add(sched, "short", 768)
    original = scheduler.Scheduler.schedule
    calls = []

    def record(self, throttle_prefills=False):
        calls.append(throttle_prefills)
        return original(self, throttle_prefills)

    monkeypatch.setattr(scheduler.Scheduler, "schedule", record)
    # Upstream allows prefills on an idle engine even when throttling is asked
    # for; admission must preserve that behavior and forward the exact flag.
    assert sched.schedule(throttle_prefills=True).num_scheduled_tokens == {"short": 768}
    assert calls == [True]
