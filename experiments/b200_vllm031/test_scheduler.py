"""Exercise the real 0.31 scheduler without loading model weights or a worker."""

import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from gleipnir.serving.monitor_score import classification_overrides

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = yaml.safe_load(
    (ROOT / "experiments/b200_inference_benchmark/config.yaml").read_text()
)
MERGED_MODEL = Path("/tmp/gleipnir-merged/fp4-full-training-bf16")
CACHED_MODEL = (
    ROOT
    / ".cache/huggingface/hub"
    / ("models--" + BENCHMARK["model"].replace("/", "--"))
    / "snapshots"
    / BENCHMARK["revision"]
)
# These scheduler tests read architecture metadata and never load model weights.
MODEL = MERGED_MODEL if (MERGED_MODEL / "config.json").is_file() else CACHED_MODEL
pytestmark = pytest.mark.skipif(
    version("vllm") != "0.31.0" or not (MODEL / "config.json").is_file(),
    reason="requires the isolated B200 0.31 runtime and pinned model config",
)


def make_scheduler(monkeypatch, *, adaptive=False, active=None, corrected=True):
    from vllm.config import PoolerConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.core.sched import scheduler
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )

    from experiments.b200_vllm031.scheduler import Vllm031PoolingScheduler

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    config = EngineArgs(
        model=str(MODEL),
        runner="pooling",
        convert="classify",
        hf_overrides=classification_overrides(
            json.loads((MODEL / "config.json").read_text())
        ),
        pooler_config=PoolerConfig(
            task="classify", pooling_type="LAST", use_activation=False
        ),
        language_model_only=True,
        max_model_len=32768,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=True,
    ).create_engine_config()
    config.cache_config.num_gpu_blocks = 4096
    config.scheduler_config.max_num_active_seqs = active
    config.scheduler_config.long_prefill_token_threshold = 128 if adaptive else 0
    config.scheduler_config.long_prefill_token_threshold_adaptive = adaptive
    binding = {
        "upstream_sha256": hashlib.sha256(
            Path(inspect.getfile(scheduler)).read_bytes()
        ).hexdigest(),
        "integration_sha256": hashlib.sha256(
            Path(inspect.getfile(Vllm031PoolingScheduler)).read_bytes()
        ).hexdigest(),
    }
    monkeypatch.setenv("GLEIPNIR_VLLM031_SCHEDULER_BINDING", json.dumps(binding))
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
            ),
        ],
    )
    cls = Vllm031PoolingScheduler if corrected else scheduler.Scheduler
    return cls(config, kv, SimpleNamespace(), 16)


def add(sched, name, length):
    from vllm.pooling_params import PoolingParams
    from vllm.v1.request import Request

    req = Request(name, [1] * length, None, PoolingParams(task="classify"))
    sched.add_request(req)
    return req


def test_exact_cap_still_requires_zero_generated_token_reservation(monkeypatch):
    old = make_scheduler(monkeypatch, corrected=False)
    req = add(old, "at_cap", 32768)
    for _ in range(40):
        old.schedule()
    assert req.num_computed_tokens == 32767
    assert old.schedule().num_scheduled_tokens == {}
    fixed = make_scheduler(monkeypatch)
    req = add(fixed, "at_cap", 32768)
    for _ in range(32):
        fixed.schedule()
    assert req.num_computed_tokens == 32768


def test_adaptive_prefill_shares_budget_and_single_request_uses_all(monkeypatch):
    sched = make_scheduler(monkeypatch, adaptive=True)
    add(sched, "long", 20000)
    add(sched, "short", 768)
    assert sched.schedule().num_scheduled_tokens == {"long": 512, "short": 512}
    alone = make_scheduler(monkeypatch, adaptive=True)
    add(alone, "long", 20000)
    assert alone.schedule().num_scheduled_tokens == {"long": 1024}


def test_active_admission_cap_keeps_runner_capacity(monkeypatch):
    sched = make_scheduler(monkeypatch, adaptive=True, active=1)
    add(sched, "first", 20000)
    add(sched, "second", 768)
    assert sched.schedule().num_scheduled_tokens == {"first": 512}
    assert sched.max_num_running_reqs == 4 and sched.max_num_active_reqs == 1
