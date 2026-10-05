"""Native specialization growth must fail the measured-update warm gate."""

import json
from types import SimpleNamespace

import pytest

from experiments.b200_mlp_gemm.grouped_kernel_audit import GroupedKernelAudit


def test_snapshot_reads_each_pinned_native_cache(monkeypatch):
    functions = {
        "fused_fwd": "tilelang_fused_chunk_gdr_fwd",
        "fused_bwd": "tilelang_fused_chunk_gdr_bwd",
        "prepare_h": "tilelang_prepare_h",
        "kkt_solve": "tilelang_kkt_solve",
        "group_reduce": "tilelang_group_reduce_vector",
    }
    calls = []

    def module(name):
        key = name.rsplit(".", 1)[-1]
        calls.append(key)
        return SimpleNamespace(
            **{
                functions[key]: SimpleNamespace(_kernel_cache={1: None, 2: None}),
            }
        )

    monkeypatch.setattr(
        "experiments.b200_mlp_gemm.grouped_kernel_audit.importlib.import_module",
        module,
    )
    assert GroupedKernelAudit.snapshot() == dict.fromkeys(functions, 2)
    assert calls == list(functions)


@pytest.mark.parametrize("step,fails", [(1, False), (10, False), (11, True)])
def test_preparation_allowed_only_before_measurement(
    tmp_path, monkeypatch, step, fails
):
    audit = GroupedKernelAudit(tmp_path)
    snapshots = iter([{"fwd": 1, "bwd": 2}, {"fwd": 2, "bwd": 2}])
    monkeypatch.setattr(audit, "snapshot", lambda: next(snapshots))
    state = SimpleNamespace(global_step=step)
    audit.on_step_begin(None, state, None)
    if fails:
        with pytest.raises(ValueError, match="prepare TileLang"):
            audit.on_step_end(None, state, None)
    else:
        audit.on_step_end(None, state, None)
    receipt = json.loads((tmp_path / "grouped_kernel_audit.json").read_text())
    assert receipt["updates"] == [{"step": step, "delta": {"fwd": 1, "bwd": 0}}]
