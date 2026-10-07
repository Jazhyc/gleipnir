"""Reproduce the exact-cap stall with the installed scheduler and CPU allocator."""

import pytest

from experiments.b200_length_admission.test_upstream import add, make_scheduler
from experiments.b200_long_context.scheduler import correct_pooling_reservation


def test_exact_cap_chunked_pooling_stalls_until_reservation_is_corrected(monkeypatch):
    sched = make_scheduler("fcfs", monkeypatch)
    req = add(sched, "at_cap", 32768)
    scheduled = 0
    for _ in range(40):
        output = sched.schedule()
        scheduled += sum(output.num_scheduled_tokens.values())
    assert scheduled == req.num_computed_tokens == 32767
    assert sched.schedule().num_scheduled_tokens == {}
    correct_pooling_reservation(sched)
    assert sched.schedule().num_scheduled_tokens == {"at_cap": 1}
    assert req.num_computed_tokens == 32768


def test_correction_rejects_generation(monkeypatch):
    sched = make_scheduler("fcfs", monkeypatch)
    sched.vllm_config.model_config.runner_type = "generate"
    with pytest.raises(ValueError, match="requires pooling"):
        correct_pooling_reservation(sched)
    assert sched.num_sampled_tokens_per_step == 1
