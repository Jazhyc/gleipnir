import pytest

from experiments.adversarial_a6_transfer.run import arm_metrics


def test_baseline_misses_are_not_transfer_evasions():
    result = arm_metrics(
        [0.1, 0.8, 0.8], [0.05, 0.1, 0.9], [0.8] * 3, [0.1] * 3, 0.2, 0.2
    )
    assert result["original_detected"] == 2
    assert result["original_missed"] == 1
    assert result["induced_evasions"] == 1
    assert result["evasion_rate_given_detected"] == 0.5
    assert result["final_missed"] == 2
    assert result["source_successes_target_initially_detected"] == 2
    assert result["source_success_transfer_rate"] == 0.5


def test_target_evasion_does_not_require_source_evasion():
    result = arm_metrics([0.8], [0.1], [0.8], [0.3], 0.2, 0.2)
    assert result["induced_evasions"] == 1
    assert result["source_successes"] == 0
    assert result["source_success_transfer_rate"] is None


def test_no_detected_cases_is_undefined_not_zero():
    result = arm_metrics([0.1], [0.3], [0.8], [0.1], 0.2, 0.2)
    assert result["evasion_rate_given_detected"] is None
    assert result["new_detections"] == 1


def test_threshold_ties_and_unchanged_choices():
    result = arm_metrics(
        [0.2, 0.3, 0.3], [0.2, 0.3, 0.2], [0.8] * 3, [0.1] * 3, 0.2, 0.2
    )
    assert result["original_detected"] == 2
    assert result["induced_evasions"] == 1
    assert result["mean_score_reduction"] == pytest.approx(1 / 30)


def test_setup_failure_still_pauses_returned_machine(tmp_path, monkeypatch):
    from experiments.adversarial_a6_transfer import orchestrate as op

    calls = []
    monkeypatch.setattr(op, "LOG", tmp_path)
    monkeypatch.setattr(op, "STATE", tmp_path / "operations.json")
    monkeypatch.setattr(op.run, "prepare", lambda: None)
    monkeypatch.setattr(
        op,
        "command",
        lambda args, **kwargs: "frozen\n" if args[1] == "rev-parse" else "",
    )

    def fake_jl(*args):
        calls.append(args)
        if args[0] == "get":
            return {"status": "Paused", "gpu_type": "A100"}
        if args[0] == "resume":
            return {"machine_id": 999}
        if args[0] == "pause":
            return {}
        raise AssertionError(args)

    monkeypatch.setattr(op, "jl", fake_jl)
    monkeypatch.setattr(
        op, "remote", lambda machine, script: "0" if script.startswith("df") else ""
    )
    with pytest.raises(AssertionError, match="30 GB"):
        op.main(123)
    assert ("pause", 999, "--yes", "--json") in calls
    assert not any(c[0] == "run" for c in calls)


def test_existing_controller_receipt_blocks_restart(tmp_path, monkeypatch):
    from experiments.adversarial_a6_transfer import orchestrate as op

    monkeypatch.setattr(op, "LOG", tmp_path)
    monkeypatch.setattr(op, "STATE", tmp_path / "operations.json")
    op.STATE.write_text("{}")
    with pytest.raises(RuntimeError, match="Prior controller"):
        op.main(123)
