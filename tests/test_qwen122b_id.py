from copy import deepcopy
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from experiments.qwen122b_id.run import build_config
from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
    optional_engine_kwargs,
    validate_config,
)


def settings():
    return OmegaConf.to_container(OmegaConf.load("experiments/qwen122b_id/config.yaml"))


def test_id_contract_does_not_inherit_ood_scope():
    config = build_config(settings(), [30_000] + [1000] * 3011)
    assert config["scope"]["rows"] == 3012
    assert config["scope"]["sources"] == ["test_stride", "gloom_exfiltration"]
    assert config["engine"]["audited_max_prompt_tokens"] == 30_000
    assert config["engine"]["audited_total_prompt_tokens"] == 3_041_000
    assert config["engine"]["canary_include_longest"]
    assert config["engine"]["canary_repeat_singletons"]
    assert optional_engine_kwargs(config) == {
        "quantization": "fp8",
        "gdn_prefill_backend": "triton",
    }


@pytest.mark.parametrize("lengths", [[1], [32768] * 3012])
def test_incomplete_or_overlong_audit_rejected(lengths):
    with pytest.raises(ValueError):
        build_config(settings(), lengths)


@pytest.mark.parametrize(
    "field,value", [("quantization", "bitsandbytes"), ("tensor_parallel_size", 1)]
)
def test_fp8_and_two_gpu_contract_required(field, value):
    config = build_config(settings(), [1000] * 3012)
    changed = deepcopy(config)
    changed["model" if field == "quantization" else "engine"][field] = value
    with pytest.raises(ValueError):
        validate_config(changed)


def test_historical_backend_defaults_unchanged():
    import json

    config = json.loads(
        Path(
            "experiments/tool_trajectory_monitoring/qwen27b_ood_benchmark.json"
        ).read_text()
    )
    assert optional_engine_kwargs(config) == {}
