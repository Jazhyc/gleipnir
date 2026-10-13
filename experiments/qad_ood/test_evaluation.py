"""Serving order and failure isolation for the fixed QAD comparison."""

import json

import numpy as np
import pytest

from experiments.filtered_fp4_ood import evaluate


def test_one_pass_does_not_claim_repeat_variability():
    assert evaluate.repeat_variation([np.asarray([0.1, 0.9])]) is None


def test_same_candidate_merge_is_reused_without_deletion(monkeypatch, tmp_path):
    from pathlib import Path

    from experiments.filtered_fp4_ood import run

    manifest = {
        "adapter_sha256": "same",
        "files_sha256": {"model.safetensors": "weights"},
    }
    monkeypatch.setattr(Path, "is_symlink", lambda self: False)
    monkeypatch.setattr(Path, "read_text", lambda self: json.dumps(manifest))
    monkeypatch.setattr(run, "file_hash", lambda path: "weights")
    monkeypatch.setattr(run.shutil, "rmtree", lambda *a: pytest.fail("must not delete"))
    writes = {}
    monkeypatch.setattr(run, "write_json", lambda p, v: writes.update({p.name: v}))
    run.restore_merge(
        tmp_path,
        {
            "teacher_merge": "/tmp/gleipnir-merged/filtered-fp4-fa4-baseline",
            "serving_adapter_sha256": "same",
            "original_merge": "original.json",
        },
        tmp_path,
    )
    assert writes["merge_identity.json"]["merge_reused"] is True


def test_serving_conditions_run_in_order_with_separate_receipts(monkeypatch, tmp_path):
    config = {"experiment": "test", "conditions": ["bf16", "optimized"]}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evaluate, "load_config", lambda: (None, config))
    seen = []

    def score(config, condition):
        seen.append(condition)
        path = tmp_path / "results/test" / condition
        path.mkdir(parents=True)
        (path / "summary.json").write_text(json.dumps({"precision": condition}))

    monkeypatch.setattr(evaluate, "evaluate_condition", score)
    evaluate.main()
    assert seen == ["bf16", "optimized"]
    result = json.loads((tmp_path / "results/test/summary.json").read_text())
    assert result["conditions"]["bf16"]["precision"] == "bf16"
    assert result["conditions"]["optimized"]["precision"] == "optimized"
    assert result["promotion"] is False


def test_failed_first_condition_stops_second(monkeypatch):
    monkeypatch.setattr(
        evaluate,
        "load_config",
        lambda: (None, {"conditions": ["bf16", "optimized"]}),
    )
    seen = []

    def fail(config, condition):
        seen.append(condition)
        raise ValueError("parity failed")

    monkeypatch.setattr(evaluate, "evaluate_condition", fail)
    with pytest.raises(ValueError, match="parity failed"):
        evaluate.main()
    assert seen == ["bf16"]
