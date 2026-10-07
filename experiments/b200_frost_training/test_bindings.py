"""CPU proofs of fixed operand order, native guard delegation and reversibility."""

from types import SimpleNamespace

import pytest

from gleipnir.kernels.fp4.frost_bindings import (
    PositionalFrostJit,
    _binding_scope,
    verify_compiler,
)


class Jit:
    def __init__(self, roles):
        self.bound = tuple(roles[::-1])
        self.original_calls = []
        self.lowered_calls = []
        self.failure = None

    def __call__(self, bindings, *, stream):
        self.original_calls.append((bindings, stream))
        return [bindings[t] for t in self.bound]

    def lowered(self, operands, *, stream):
        if self.failure:
            raise self.failure
        self.lowered_calls.append((operands, stream))
        return operands


def plan_type():
    # Fresh class per test avoids scope/method state leaking between tests.
    class Plan:
        def __init__(self):
            roles = [object() for _ in range(6)]
            for name, role in zip(
                [
                    "a_tensor",
                    "b_tensor",
                    "sfa_tensor",
                    "sfb_tensor",
                    "scale_tensor",
                    "output_tensor",
                ],
                roles,
                strict=True,
            ):
                setattr(self, name, role)
            self.jit = Jit(roles)
            self.workspace_bytes = 0
            self.plan = SimpleNamespace(k=2560, n=9216)

        def __call__(self, a, b):
            return self.jit(
                dict(
                    zip(
                        [
                            self.a_tensor,
                            self.b_tensor,
                            self.sfa_tensor,
                            self.sfb_tensor,
                            self.scale_tensor,
                            self.output_tensor,
                        ],
                        [a, b, "scale-a", "scale-b", "descale", object()],
                        strict=True,
                    )
                ),
                stream=17,
            )

    return Plan


def test_order_fresh_operands_and_native_guards():
    p = plan_type()()
    original = p.jit
    wrapper = PositionalFrostJit(p)
    values = [object() for _ in range(6)]
    mapping = dict(zip(original.bound, values, strict=True))
    assert wrapper(mapping, stream=42) == values
    assert original.lowered_calls[-1][1] == 42
    assert original.original_calls == []
    changed = {tensor: object() for tensor in original.bound}
    assert wrapper(changed, stream=99) == [changed[t] for t in original.bound]
    assert wrapper.bound is original.bound
    original.failure = RuntimeError("native alignment guard")
    with pytest.raises(RuntimeError, match="native alignment guard"):
        wrapper(mapping, stream=42)
    assert wrapper.calls == 2


@pytest.mark.parametrize(
    "mutation", ["extra", "duplicate", "missing", "workspace", "unlowered"]
)
def test_unsupported_plan_fails_closed(mutation):
    p = plan_type()()
    if mutation == "extra":
        p.jit.bound = (*p.jit.bound, object())
    if mutation == "duplicate":
        p.jit.bound = (p.jit.bound[0],) * 6
    if mutation == "missing":
        p.sfa_tensor = object()
    if mutation == "workspace":
        p.workspace_bytes = 1
    if mutation == "unlowered":
        p.jit.lowered = None
    with pytest.raises(ValueError, match="six-tensor|zero-workspace"):
        PositionalFrostJit(p)


def test_mapping_rejects_missing_extra_and_foreign_tensor():
    p = plan_type()()
    proxy = PositionalFrostJit(p)
    good = {t: object() for t in p.jit.bound}
    for mapping in [dict(list(good.items())[:-1]), {**good, object(): object()}]:
        with pytest.raises(ValueError, match="exactly six"):
            proxy(mapping, stream=1)
    bad = good.copy()
    del bad[p.jit.bound[0]]
    bad[object()] = object()
    with pytest.raises(ValueError, match="unknown graph tensor"):
        proxy(bad, stream=1)


def test_modes_use_unwrapped_control_and_restore_old_and_new_plans():
    Plan = plan_type()
    original_method = Plan.__call__
    first = Plan()
    original_jit = first.jit
    with _binding_scope(Plan, "original") as control:
        assert Plan.__call__ is original_method
        first("a", "b")
        assert control.state()["plans"] == 0
        control.set_mode("direct")
        first("a", "b")
        new = Plan()
        new_original = new.jit
        new("c", "d")
        assert control.state()["direct_calls"] == 2
        assert len(original_jit.original_calls) == 1
        assert len(original_jit.lowered_calls) == 1
        control.set_mode("original")
        assert Plan.__call__ is original_method
        assert first.jit is original_jit and new.jit is new_original
        first("a2", "b2")
        assert len(original_jit.original_calls) == 2
        control.set_mode("direct")
        first("a3", "b3")
        assert control.state()["direct_calls"] == 3
    assert first.jit is original_jit and new.jit is new_original
    assert Plan.__call__ is original_method
    assert control.state()["closed"]
    with pytest.raises(RuntimeError, match="closed"):
        control.set_mode("direct")


def test_failures_restore_scope_and_nested_scopes_are_rejected():
    Plan = plan_type()
    method = Plan.__call__
    p = Plan()
    jit = p.jit
    with pytest.raises(RuntimeError, match="native guard"):
        with _binding_scope(Plan, "direct"):
            with pytest.raises(RuntimeError, match="already installed"):
                with _binding_scope(Plan, "direct"):
                    pass
            jit.failure = RuntimeError("native guard")
            p("a", "b")
    assert Plan.__call__ is method and p.jit is jit
    with pytest.raises(ValueError, match="unknown.*mode"):
        with _binding_scope(Plan, "bad"):
            pass
    assert Plan.__call__ is method
    with _binding_scope(Plan, "original"):
        pass


def test_compiler_drift_is_rejected_before_dispatch(tmp_path):
    source = tmp_path / "compiler.py"
    source.write_text("unknown vendor revision")
    with pytest.raises(ValueError, match="compiler changed"):
        verify_compiler(source)


def test_executor_drift_restores_every_other_plan_before_reporting_error():
    Plan = plan_type()
    method = Plan.__call__
    first, second = Plan(), Plan()
    second_jit = second.jit
    replacement = object()
    with pytest.raises(RuntimeError, match="executor changed"):
        with _binding_scope(Plan, "direct") as control:
            first("a", "b")
            second("c", "d")
            first.jit = replacement
    assert first.jit is replacement
    assert second.jit is second_jit
    assert Plan.__call__ is method
    assert control.closed


def test_probe_gate_requires_every_gradient_replay_and_geometry():
    from experiments.b200_frost_training.probe import GEOMETRIES, accept_checks

    checks = {
        key: True
        for key in [
            "output_exact",
            "gradients_exact",
            "finite",
            "independent_outputs",
            "changed_input_replay_exact",
            "changed_input_replay_changed",
            "changed_master_replay_exact",
            "changed_master_replay_changed",
            "alternate_stream_exact",
        ]
    }
    checks.update(
        gradient_count=7,
        dispatch={"geometries": [dict(k=k, n=n, calls=1) for k, n in GEOMETRIES]},
    )
    accept_checks(checks)
    for key in list(checks):
        if isinstance(checks[key], bool):
            with pytest.raises(ValueError, match="parity failed"):
                accept_checks({**checks, key: False})
    with pytest.raises(ValueError, match="parity failed"):
        accept_checks({**checks, "gradient_count": 6})
    checks["dispatch"]["geometries"].pop()
    with pytest.raises(ValueError, match="coverage incomplete"):
        accept_checks(checks)


@pytest.mark.parametrize(
    "exit_race,timeout", [(False, False), (True, False), (False, True)]
)
def test_timeout_stops_only_probe_group_and_handles_exit_races(
    monkeypatch, exit_race, timeout
):
    import subprocess

    from experiments.b200_frost_training import probe

    calls = []

    def kill(pid, sig):
        calls.append(("kill", pid, sig))
        if exit_race:
            raise ProcessLookupError()

    waits = []

    def wait(**kwargs):
        waits.append(kwargs)
        if timeout and len(waits) == 1:
            raise subprocess.TimeoutExpired("probe", 5)
        return 0

    monkeypatch.setattr(probe.os, "killpg", kill)
    probe.stop_worker(SimpleNamespace(pid=9876, wait=wait))
    assert all(call[1] == 9876 for call in calls)
    assert calls[0][2] == probe.signal.SIGTERM
    if timeout:
        assert calls[1][2] == probe.signal.SIGKILL
    assert waits[-1] == ({} if (timeout or exit_race) else {"timeout": 5})


def test_probe_refuses_another_gpu_user_without_stopping_it():
    from experiments.b200_frost_training.probe import require_exclusive_gpu

    require_exclusive_gpu("", 123)
    require_exclusive_gpu("123\n", 123)
    with pytest.raises(RuntimeError, match="another process"):
        require_exclusive_gpu("123\n456\n", 123)
