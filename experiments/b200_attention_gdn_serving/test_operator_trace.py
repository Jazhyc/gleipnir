"""Check that diagnostic evidence survives failure without changing results."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from gleipnir.serving_operator_trace import OperatorTrace, PatchSet


def events(path):
    return [
        json.loads(line) for line in (path / "operators.jsonl").read_text().splitlines()
    ]


def test_begin_visible_inside_operator_and_result_preserved(tmp_path):
    trace = OperatorTrace(
        tmp_path, "launch", lambda: pytest.fail("unexpected sync"), lambda: False
    )
    result = object()

    def operator():
        assert events(tmp_path)[-1]["phase"] == "begin"
        return result

    with trace.batch():
        assert trace.invoke("kernel", operator) is result
    assert (
        next(e for e in events(tmp_path) if e["phase"] == "end")["gpu_completed"]
        is False
    )


def test_failed_synchronization_retains_pending_operator(tmp_path):
    error = RuntimeError("GPU failure")

    def synchronize():
        assert events(tmp_path)[-1]["phase"] == "synchronize"
        raise error

    trace = OperatorTrace(tmp_path, "sync", synchronize, lambda: False)
    with trace.batch(), pytest.raises(RuntimeError) as caught:
        trace.invoke("gdn", lambda: None, rows=32767)
    assert caught.value is error
    records = events(tmp_path)
    assert not any(e["phase"] == "end" for e in records)
    assert next(e for e in records if e["phase"] == "error")["operator"] == "gdn"


def test_capture_and_startup_do_not_synchronize(tmp_path):
    trace = OperatorTrace(
        tmp_path, "sync", lambda: pytest.fail("capture sync"), lambda: True
    )
    assert trace.invoke("startup", lambda: 1) == 1
    with trace.batch():
        assert trace.invoke("capture", lambda: 2) == 2
    assert not any(e["phase"] == "begin" for e in events(tmp_path))


def test_batch_snapshot_records_exact_order_and_checksum(tmp_path):
    trace = OperatorTrace(tmp_path, "launch", lambda: None, lambda: False)
    with trace.batch():
        path = trace.save_inputs(["second", "first"], [[4, 7, 2], [9]])
    record = next(e for e in events(tmp_path) if e["phase"] == "inputs")
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(path.read_text()) == {
        "request_ids": ["second", "first"],
        "token_ids": [[4, 7, 2], [9]],
    }
    assert record["lengths"] == [3, 1]
    with pytest.raises(ValueError):
        trace.save_inputs(["one"], [[], [2]])


def test_restore_removes_diagnostic_wrappers_and_restores_inheritance():
    class Base:
        def run(self):
            return "original"

    obj = Base()
    namespace = SimpleNamespace(run=lambda: "native")
    native = namespace.run
    patches = PatchSet()
    patches.set(obj, "run", lambda: "diagnostic")
    patches.set(namespace, "run", lambda: "diagnostic")
    patches.restore()
    assert "run" not in vars(obj)
    assert obj.run() == "original"
    assert namespace.run is native
