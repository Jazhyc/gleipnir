"""CPU admission ordering, aging, queue lifecycle and serving contracts."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sys
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from experiments.b200_length_admission.run import (
    INTEGRATION_SOURCE,
    POLICY_SOURCE,
    SCHEDULER,
    admission_command,
    policy_config,
    process_is_live,
    require_idle_gpu,
)
from experiments.b200_length_admission.traffic import arrival_offsets, replay
from gleipnir.serving.length_admission import AdmissionPolicy, LengthAdmissionMixin
from gleipnir.serving.score_runtime import resume_score_environment

ROOT = Path(__file__).resolve().parents[2]


def request(name, length, arrival=100.0, computed=0, blocked=False):
    return SimpleNamespace(
        request_id=name,
        num_prompt_tokens=length,
        arrival_time=arrival,
        num_computed_tokens=computed,
        blocked=blocked,
    )


class Queue(deque):
    def peek_request(self):
        return self[0]

    def pop_request(self):
        return self.popleft()

    def prepend_request(self, req):
        self.appendleft(req)

    def remove_requests(self, requests):
        for req in requests:
            self.remove(req)


class BaseScheduler:
    def _select_waiting_queue_for_scheduling(self):
        return self.skipped_waiting or self.waiting or None

    def schedule(self, defer_prefills=False):
        self.defer_prefills = defer_prefills
        return self._select_waiting_queue_for_scheduling()


class QueuedScheduler(LengthAdmissionMixin, BaseScheduler):
    def __init__(self, waiting=(), skipped=(), policy=None):
        self.waiting = Queue(waiting)
        self.skipped_waiting = Queue(skipped)
        self.running = [request("running", 16000)]
        self.admission_policy = policy or AdmissionPolicy()

    def drain(self):
        admitted, blocked = [], []
        while queue := self._select_waiting_queue_for_scheduling():
            req = queue.pop_request()
            (blocked if req.blocked else admitted).append(req.request_id)
        return admitted, blocked


def test_buckets_are_fifo_within_bucket_not_shortest_first():
    policy = AdmissionPolicy()
    rows = [
        request("long", 25000),
        request("first_short", 1000),
        request("tiny_later", 188, 100.01),
        request("medium", 4096),
    ]
    sched = QueuedScheduler(rows)
    sched.order_waiting(100.1)
    assert sched.drain()[0] == ["first_short", "tiny_later", "medium", "long"]
    boundaries = [1024, 1025, 4096, 4097, 16384, 16385, 32768, 32769]
    assert [policy.key(request(str(n), n), 100.1)[1] for n in boundaries] == [
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
    ]


def test_aged_long_request_precedes_continuous_new_short_arrivals():
    old = request("old", 28733)
    sched = QueuedScheduler([old, request("short", 200, 100.2)])
    sched.order_waiting(100.249)
    assert sched.waiting.peek_request().request_id == "short"
    sched.waiting.append(request("newest", 188, 100.25))
    sched.order_waiting(100.25)
    assert sched.waiting.pop_request() is old
    assert sched.drain()[0] == ["short", "newest"]


def test_both_queues_aging_blocking_and_running_preservation():
    rows = [request("short", 200, 100.2), request("old_wait", 27000, 99.0)]
    skipped = [
        request("blocked", 500, 98.0, blocked=True),
        request("old_skip", 26000, 99.1),
        request("fresh_skip", 4000, 100.2),
    ]
    sched = QueuedScheduler(rows, skipped)
    running = sched.running.copy()
    sched.order_waiting(100.3)
    admitted, blocked = sched.drain()
    assert blocked == ["blocked"]
    assert admitted == ["old_wait", "old_skip", "short", "fresh_skip"]
    assert sched.running == running
    assert sched._select_waiting_queue_for_scheduling() is None


def test_preemption_cancellation_equal_arrivals_and_remaining_work():
    a, b = request("a", 1000), request("b", 300)
    sched = QueuedScheduler([a, b, request("cancelled", 200)])
    sched.waiting.remove_requests([sched.waiting[-1]])
    sched.waiting.prepend_request(request("preempted", 20000, 99.0))
    sched.order_waiting(100.1)
    assert sched.drain()[0] == ["preempted", "a", "b"]
    assert AdmissionPolicy().key(request("resumed", 5000, computed=4500), 100.1)[1] == 0


def test_fcfs_control_retains_upstream_prepend_and_skipped_precedence():
    sched = QueuedScheduler(
        [request("long", 25000), request("short", 200)],
        [request("skipped", 4000)],
        AdmissionPolicy(mode="fcfs"),
    )
    sched.order_waiting(100.5)
    assert sched.drain()[0] == ["skipped", "long", "short"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_wait_seconds": 0},
        {"max_wait_seconds": float("nan")},
        {"length_buckets": ()},
        {"length_buckets": (4096, 1024)},
        {"length_buckets": (1024, 1024)},
        {"length_buckets": (True,)},
        {"mode": "priority"},
    ],
)
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        AdmissionPolicy(**kwargs)


def settings():
    return json.loads(
        (ROOT / "experiments/b200_length_admission/config.json").read_text()
    )


def parent_command():
    return [
        "python",
        "-m",
        "experiments.b200_mutation_analysis.server",
        "--runner",
        "pooling",
        "--worker-cls",
        "selected.worker",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--max-num-batched-tokens",
        "32768",
        "--max-num-seqs",
        "128",
        "--additional-config",
        json.dumps({"monitor_score": {}, "qk_mutation_analysis": True}),
    ]


def test_launch_changes_only_admission_and_binds_sources():
    parent = parent_command()
    command = admission_command(
        parent, settings(), {POLICY_SOURCE: "a", INTEGRATION_SOURCE: "b"}
    )
    assert "--scheduler-cls" not in parent
    assert command[command.index("--scheduler-cls") + 1] == SCHEDULER
    assert "--no-async-scheduling" in command
    for flag in (
        "--runner",
        "--worker-cls",
        "--max-num-seqs",
        "--max-num-batched-tokens",
    ):
        assert command[command.index(flag) + 1] == parent[parent.index(flag) + 1]
    additional = json.loads(command[command.index("--additional-config") + 1])
    assert additional["monitor_score"] == {}
    assert "length_admission" not in additional
    config = policy_config(settings(), {POLICY_SOURCE: "a", INTEGRATION_SOURCE: "b"})
    assert config["policy_sha256"] == "a"
    assert config["max_wait_seconds"] == 0.25
    for flag in ("--scheduler-cls", "--async-scheduling"):
        with pytest.raises(ValueError, match="already overrides"):
            admission_command(parent + [flag, "other"], settings(), {})


def test_idle_launch_never_reclaims_an_occupied_gpu(monkeypatch):
    monkeypatch.setattr("subprocess.check_output", lambda *a, **kw: "1234\n")
    with pytest.raises(ValueError, match="occupied"):
        require_idle_gpu()
    require_idle_gpu(allowed_pids={1234})
    with pytest.raises(ValueError, match="occupied"):
        require_idle_gpu(allowed_pids={5678})
    monkeypatch.setattr("subprocess.check_output", lambda *a, **kw: "")
    require_idle_gpu()


def test_mixed_control_changes_only_policy_mode():
    original = settings()
    hashes = {POLICY_SOURCE: "a", INTEGRATION_SOURCE: "b"}
    candidate = admission_command(parent_command(), original, hashes)
    control = admission_command(parent_command(), original, hashes, control_only=True)
    assert control == candidate
    control_config = policy_config(original, hashes, control_only=True)
    candidate_config = policy_config(original, hashes)
    assert control_config["mode"] == "fcfs"
    control_config["mode"] = "length_aware"
    assert control_config == candidate_config
    assert original["length_admission"]["mode"] == "length_aware"


def test_exited_zombie_is_not_sent_to_live_stop_helper(tmp_path):
    pid = tmp_path / "1234"
    pid.mkdir()
    stat = pid / "stat"
    stat.write_text("1234 (API server) Z 1 2 3")
    assert not process_is_live(1234, tmp_path)
    stat.write_text("1234 (API server) S 1 2 3")
    assert process_is_live(1234, tmp_path)
    assert not process_is_live(9999, tmp_path)


def test_retired_runtime_preserves_identity_caches_and_parent(tmp_path, monkeypatch):
    selection = tmp_path / "experiments/b200_inference_benchmark/baseline.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(json.dumps({"mutation_validation": "results/mutation.json"}))
    frontend_calls = []
    helpers = {
        "gleipnir.training.backends.native_fp4": {
            "native_fp4_environment": lambda env, root: env,
        },
        "gleipnir.training.backends.qwen35": {
            "DEFAULT_TRITON_TARGET": Path("/tmp/triton"),
            "triton_environment": lambda target, env: env,
        },
        "gleipnir.serving.runtime": {
            "local_serving_runtime": lambda root, env: {
                "manifest_sha256": "runtime",
                "python": "python",
            },
        },
        "gleipnir.serving.gigatoken": {
            "configure_frontend": lambda *args: frontend_calls.append(args),
        },
    }
    for name, attributes in helpers.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    parent = {
        "command": parent_command(),
        "local_runtime": {"manifest_sha256": "runtime"},
        "frontend": {"backend": "gigatoken_native"},
        "host_wrapper": {"validation": "results/wrapper.json"},
        "cache_paths": {"VLLM_CACHE_ROOT": "/persistent/cache", "unrelated": "skip"},
    }
    base = {"PYTHONPATH": "src:."}
    env = resume_score_environment(tmp_path, parent, base)
    assert base == {"PYTHONPATH": "src:."}
    assert env["VLLM_CACHE_ROOT"] == "/persistent/cache" and "unrelated" not in env
    assert env["GLEIPNIR_STRIDE_VALIDATION"] == "results/mutation.json"
    assert parent["command"][2] == "experiments.b200_mutation_analysis.server"
    assert frontend_calls[0][2][2] == "experiments.b200_attention_gdn_serving.server"
    parent["local_runtime"]["manifest_sha256"] = "drift"
    with pytest.raises(ValueError, match="runtime identity changed"):
        resume_score_environment(tmp_path, parent, base)


def test_vllm_adapter_composes_upstream_schedule_and_rejects_source_drift(
    tmp_path,
    monkeypatch,
):
    # Load the adapter with CPU doubles for vLLM's GPU-dependent import graph.
    sched_module, queue_module = ModuleType("scheduler"), ModuleType("request_queue")
    for module in (sched_module, queue_module):
        source = tmp_path / f"{module.__name__}.py"
        source.write_text("# pinned source\n")
        module.__file__ = str(source)

    class Upstream(BaseScheduler):
        def __init__(self, config):
            self.vllm_config = config
            self.scheduler_config = config.scheduler_config
            self.waiting = Queue([request("long", 20000), request("short", 200)])
            self.skipped_waiting = Queue()

    sched_module.Scheduler = Upstream
    package = ModuleType("vllm.v1.core.sched")
    package.scheduler, package.request_queue = sched_module, queue_module
    logger = ModuleType("vllm.logger")
    logger.init_logger = lambda _: SimpleNamespace(info=lambda *a: None)
    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched", package)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    path = ROOT / INTEGRATION_SOURCE
    spec = importlib.util.spec_from_file_location("cpu_length_scheduler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "version", lambda _: "0.24.0")
    monkeypatch.setattr(module.time, "time", lambda: 100.1)
    config = settings()["length_admission"].copy()
    config.update(
        scheduler_sha256=hashlib.sha256(
            Path(sched_module.__file__).read_bytes()
        ).hexdigest(),
        request_queue_sha256=hashlib.sha256(
            Path(queue_module.__file__).read_bytes()
        ).hexdigest(),
        policy_sha256=hashlib.sha256((ROOT / POLICY_SOURCE).read_bytes()).hexdigest(),
        integration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    vllm_config = SimpleNamespace(
        additional_config={},
        scheduler_config=SimpleNamespace(
            async_scheduling=False, policy="fcfs", enable_chunked_prefill=True
        ),
        model_config=SimpleNamespace(runner_type="pooling"),
    )
    monkeypatch.setenv("GLEIPNIR_LENGTH_ADMISSION_CONFIG", json.dumps(config))
    instance = module.LengthAwareScheduler(vllm_config)
    assert instance.schedule(True).peek_request().request_id == "short"
    assert instance.defer_prefills is True
    config["scheduler_sha256"] = "drift"
    monkeypatch.setenv("GLEIPNIR_LENGTH_ADMISSION_CONFIG", json.dumps(config))
    with pytest.raises(ValueError, match="upstream source drift"):
        module.LengthAwareScheduler(vllm_config)
    vllm_config.scheduler_config.async_scheduling = True
    with pytest.raises(ValueError, match="synchronous"):
        module.LengthAwareScheduler(vllm_config)


def test_arrivals_are_reproducible_rate_scaled_and_finite():
    a, b = arrival_offsets(320, 40, 17), arrival_offsets(320, 80, 17)
    assert a == arrival_offsets(320, 40, 17)
    assert a[0] == 0 and all(x < y for x, y in zip(a, a[1:], strict=False))
    assert a == [2 * x for x in b]
    for rate in (0, -1, float("inf")):
        with pytest.raises(ValueError):
            arrival_offsets(320, rate, 17)


def test_traffic_preserves_identity_and_measures_arrival_to_response(monkeypatch):
    real_client = httpx.AsyncClient

    def handler(req):
        assert json.loads(req.content)["prompt"] == "hello"
        return httpx.Response(
            200,
            json={
                "score": 0.5,
                "margin": 0.0,
                "logits": [0.0, 0.0],
                "prompt_tokens": 2,
            },
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(**kw, transport=httpx.MockTransport(handler)),
    )
    rows = [{"id": "a", "prompt": "hello", "prompt_tokens": 2, "prompt_sha256": "hash"}]
    values, seconds = asyncio.run(replay(rows, [0.0], 8010))
    assert values[0]["id"] == "a" and values[0]["prompt_sha256"] == "hash"
    assert values[0]["latency_seconds"] >= values[0]["http_latency_seconds"] >= 0
    assert seconds >= values[0]["latency_seconds"]
