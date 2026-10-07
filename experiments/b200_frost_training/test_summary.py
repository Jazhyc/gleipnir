"""Complete-update reporting must exclude warmup and preserve all timed samples."""

import json

import pytest

from experiments.b200_frost_training.summarize_resident import summarize


def test_complete_update_summary_and_drift_gate(tmp_path):
    for name in ("01baseline", "02repeat", "03direct", "04control"):
        trial = tmp_path / name
        trial.mkdir()
        receipt = {
            "status": "complete",
            "step_seconds": [99.0] * 10 + [1.0 if name == "03direct" else 2.0] * 10,
            "physical_contract": [{"update": n, "tokens": 100} for n in range(1, 21)],
            "initial_master_sha256": "initial",
            "final_master_sha256": "final",
            "loss_history": [1.0],
            "pid": 42,
            "measured_updates_warm": True,
            "instrumented": False,
            "optimizer_reset": True,
        }
        (trial / "receipt.json").write_text(json.dumps(receipt))
    (tmp_path / "03direct/candidate_validation.json").write_text(
        json.dumps(
            {
                "accepted_for_timing": True,
                "direct_host_calls_in_gate": 2,
                "binding_state": {"direct_calls": 2},
            }
        )
    )
    (tmp_path / "03direct/candidate_execution.json").write_text(
        json.dumps({"direct_calls": 9})
    )
    result = summarize(tmp_path)
    assert result["relative_time_reduction"] == 0.5
    assert result["trials"]["03direct"]["actual_tokens_per_second"] == 100
    assert result["direct_host_calls_in_training"] == 7
    assert result["eligible_for_selection"]
    path = tmp_path / "03direct/receipt.json"
    receipt = json.loads(path.read_text())
    receipt["final_master_sha256"] = "changed"
    path.write_text(json.dumps(receipt))
    assert not summarize(tmp_path)["eligible_for_selection"]
    receipt["step_seconds"].pop()
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="incomplete/nonfinite"):
        summarize(tmp_path)
