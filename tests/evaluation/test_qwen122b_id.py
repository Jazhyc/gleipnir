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


def test_supervisor_rejects_busy_gpus(tmp_path, monkeypatch):
    from experiments.qwen122b_id import run as runner

    (tmp_path / "manifest.json").write_text('{"files": {}}')
    monkeypatch.setattr(runner, "gpu_memory", lambda: [70_000, 0])
    with pytest.raises(RuntimeError, match="must be idle"):
        runner.run(tmp_path)


@pytest.mark.parametrize("returncode,expected", [(0, "complete"), (1, "failed")])
def test_supervisor_records_child_outcome(tmp_path, monkeypatch, returncode, expected):
    import json

    from experiments.qwen122b_id import run as runner

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            assert kwargs["start_new_session"] is True
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
            self.returncode = returncode

        def poll(self):
            return self.returncode

    monkeypatch.chdir(tmp_path)
    (tmp_path / "manifest.json").write_text('{"files": {}}')
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation/result.json").write_text('{"rows": 3012}')
    monkeypatch.setattr(runner, "gpu_memory", lambda: [0, 0])
    monkeypatch.setattr(runner.subprocess, "Popen", FakeProcess)
    if returncode:
        with pytest.raises(RuntimeError, match="exited"):
            runner.run(tmp_path)
    else:
        runner.run(tmp_path)
    assert json.loads((tmp_path / "status.json").read_text())["state"] == expected


@pytest.mark.parametrize("version", ["35", "38"])
def test_followups_compose_same_scoring_and_engine_contract(version):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        version_base=None, config_dir=str(Path("experiments/qwen122b_id").resolve())
    ):
        followup = OmegaConf.to_container(
            compose(config_name=f"qwen{version}_27b"), resolve=True
        )
    config = build_config(followup, [1000] * 3012)
    assert config["model"]["id"] == f"Qwen/Qwen3.{version[-1]}-27B-FP8"
    assert config["model"]["quantization"] == "fp8"
    assert config["engine"] == build_config(settings(), [1000] * 3012)["engine"]
    assert followup["result_dir"] == f"results/qwen{version}_27b_id"


@pytest.mark.parametrize(
    "state,rows,ready", [("running", 0, False), ("complete", 3012, True)]
)
def test_dependency_waits_for_complete_coverage(tmp_path, state, rows, ready):
    import json

    from experiments.qwen122b_id.run import dependency_ready

    (tmp_path / "status.json").write_text(json.dumps({"state": state}))
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation/result.json").write_text(json.dumps({"rows": rows}))
    assert dependency_ready(tmp_path) is ready


def test_failed_predecessor_is_not_treated_as_gpu_release(tmp_path):
    from experiments.qwen122b_id.run import dependency_ready

    (tmp_path / "status.json").write_text('{"state": "failed"}')
    with pytest.raises(RuntimeError, match="predecessor failed"):
        dependency_ready(tmp_path)
