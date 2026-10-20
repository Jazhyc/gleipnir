from pathlib import Path

import pytest

from experiments.qad_v02.generate import generate, transform

ROOT = Path(__file__).resolve().parents[2]
VARIANT = {
    "config_path": "experiments/qad_v02/config.yaml",
    "runtime_module": "experiments.qad_v02.runtime",
    "native_attention_backward": False,
}


def test_stages_point_only_at_v02_receipts(tmp_path) -> None:
    stages = generate(ROOT, tmp_path, **VARIANT)
    train = stages["train"].read_text()
    reference = stages["reference"].read_text()
    for source in (train, reference):
        assert "filtered-fp4" not in source
        assert "experiments/qad_v02/config.yaml" in source
        assert 'config["canary_reference"]' in source
        assert 'config["teacher_experiment"]' in source
    assert train.count('config["initial_server_canary"]') == 2
    assert "from experiments.qad_v02.runtime import attention_runtime" in train


def test_template_drift_fails_closed() -> None:
    original = (ROOT / "experiments/quantization_recovery/train.py").read_text()
    with pytest.raises(ValueError):
        transform(
            original.replace("canary_repeat1.json", "canary.json"), "train", **VARIANT
        )
