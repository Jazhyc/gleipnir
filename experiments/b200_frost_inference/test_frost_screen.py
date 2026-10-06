"""Protect training runtime/source identity and frozen serving choices."""

import json

import pytest

from experiments.b200_frost_inference.run import EXPERIMENT, resolve_condition


def test_frost_uses_training_runtime_and_preserves_attention_backend():
    condition = json.loads((EXPERIMENT / "config.json").read_text())
    resolved = resolve_condition(condition, {"kernel": "first"})
    args = resolved["extra_server_args"]
    assert args[args.index("--attention-backend") + 1] == "FLASHINFER"
    assert "--linear-backend" not in args
    assert (
        resolve_condition(condition, {"kernel": "second"})["extra_server_args"] != args
    )
    with pytest.raises(ValueError, match="training-forward"):
        resolve_condition({**condition, "training_fp4_environment": False}, {})
