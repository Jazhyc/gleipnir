"""Protect the native MLP-only precision and idle-control launch contract."""

import json
import re

import pytest
import yaml

from experiments.b200_inference_benchmark.run import server_command
from experiments.b200_inference_kernels.run import BENCHMARK, EXPERIMENT, condition_args


def test_native_condition_preserves_attention_head_and_control():
    condition = json.loads((EXPERIMENT / "config.json").read_text())
    args = condition_args(condition)
    assert args[args.index("--quantization") + 1] == "fp8_per_channel"
    rule = condition["quantization_config"]["ignore"][0].removeprefix("re:")
    assert re.match(rule, "model.language_model.layers.0.mlp.gate_proj") is None
    assert re.match(rule, "model.language_model.layers.31.mlp.down_proj") is None
    for layer in [
        "model.language_model.layers.0.linear_attn.in_proj_qkvz",
        "model.language_model.layers.3.self_attn.q_proj",
        "lm_head",
        "visual.proj",
    ]:
        assert re.match(rule, layer)
    config = yaml.safe_load((BENCHMARK / "config.yaml").read_text())
    config["port"] = condition["port"]
    from pathlib import Path

    command = server_command(config, Path(condition["merged_model"])) + args
    assert command[command.index("--port") + 1] == str(condition["port"])
    assert "--enable-lora" not in command
    assert command[command.index("--max-num-seqs") + 1] == "16"


def test_rejects_accidental_scope_or_endpoint_change():
    condition = json.loads((EXPERIMENT / "config.json").read_text())
    with pytest.raises(ValueError, match="scope drift"):
        condition_args({**condition, "quantization_config": {"ignore": []}})
    with pytest.raises(ValueError, match="control endpoint"):
        condition_args({**condition, "port": 8000})
