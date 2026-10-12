"""Direct Kimi stages preserve privilege and keep final serving gates strict."""

import json
from pathlib import Path

import pytest

from experiments.kimi_direct_qat import runtime
from experiments.kimi_direct_qat.generate import transform


@pytest.mark.parametrize("stage", ["train", "reference", "evaluate"])
def test_direct_stages_preserve_declared_derivative_boundaries(stage):
    source = Path(f"experiments/quantization_recovery/{stage}.py").read_text()
    generated, changes = transform(
        source,
        stage,
        config_path="experiments/kimi_direct_qat/config.yaml",
        runtime_module="experiments.kimi_direct_qat.runtime",
    )
    restored = generated
    for change in reversed(changes):
        assert generated.count(change["new"]) == change["count"]
        restored = restored.replace(change["new"], change["old"])
    assert restored == source
    if stage == "train":
        assert 'directory / "teacher_complete.json"' not in generated
        assert "load_teachers(workload, root, output, config)" in generated
        assert 'config["initial_alignment"]' in generated
    elif stage == "evaluate":
        assert 'config["initial_alignment"]' not in generated
        assert 'config["alignment"]' in generated


@pytest.mark.parametrize("disagree", [False, True])
def test_target_join_uses_kimi_privilege_and_enforces_strict_filter(
    monkeypatch, tmp_path, disagree
):
    rows = [{"label": 1}, {"label": 0}]
    config = {"teacher_cache": "original/kimi.jsonl", "teacher_sha256": "frozen"}
    receipt = {
        "cache_sha256": "frozen",
        "teacher_model": "moonshotai/kimi-k3",
        "prompt_contract": "privileged",
        "bindings": [
            {"teacher_prompt_sha256": "teacher", "student_prompt_sha256": "student"}
        ],
    }

    def loader(workload, cache, **kwargs):
        assert workload is rows
        assert cache == tmp_path / config["teacher_cache"]
        assert kwargs == {
            "expected_sha256": "frozen",
            "teacher_model": "moonshotai/kimi-k3",
            "prompt_contract": "privileged",
        }
        return ([0.2, 0.2] if disagree else [0.8, 0.2]), receipt

    monkeypatch.setattr(runtime, "load_binary_targets", loader)
    if disagree:
        with pytest.raises(ValueError, match="agreement"):
            runtime.load_teachers(rows, tmp_path, tmp_path, config)
    else:
        targets, binding = runtime.load_teachers(rows, tmp_path, tmp_path, config)
        assert targets == [{"soft_target": 0.8}, {"soft_target": 0.2}]
        assert binding["prompt_contract"] == "privileged"
        assert (
            json.loads((tmp_path / "kimi_target_identity.json").read_text()) == receipt
        )


def test_missing_teacher_block_fails_closed():
    source = (
        Path("experiments/quantization_recovery/train.py")
        .read_text()
        .replace('directory / "teacher_complete.json"', 'directory / "changed.json"')
    )
    with pytest.raises(ValueError, match="teacher loading template"):
        transform(source, "train")


@pytest.mark.parametrize("stage", ["train", "evaluate"])
def test_initial_diagnostic_exception_does_not_change_final_serving_gate(stage):
    source = Path(f"experiments/quantization_recovery/{stage}.py").read_text()
    generated, changes = transform(
        source, stage, initial_alignment_diagnostic_only=True
    )
    if stage == "train":
        assert "untrained_base_alignment_diagnostic_failed" in generated
        assert 'raise ValueError("QAD packed-sequence isolation failed")' in generated
    else:
        assert 'raise ValueError("new QAD master/serving parity failed")' in generated
        assert "untrained_base_alignment_diagnostic_failed" not in generated
    for change in reversed(changes):
        generated = generated.replace(change["new"], change["old"])
    assert generated == source
