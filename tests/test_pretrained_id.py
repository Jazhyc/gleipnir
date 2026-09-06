import hashlib
import json
from pathlib import Path

import pytest

from experiments.pretrained_id.run import resolve_config
from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import validate_inputs
from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import validate_config


@pytest.mark.parametrize("size", ["4b", "9b"])
def test_pretrained_contract(size):
    config = resolve_config(Path("experiments/pretrained_id/config.yaml"), size)
    assert config["model"]["id"] == f"Qwen/Qwen3.5-{size.upper()}-Base"
    assert len(config["model"]["revision"]) == 40
    assert config["prompt"]["role"] == "student"
    assert config["scope"]["rows"] == 3012
    assert config["engine"]["tensor_parallel_size"] == 1
    assert config["engine"]["canary_repeat_singletons"]
    config["prompt"]["role"] = "unknown"
    with pytest.raises(ValueError, match="declare"):
        validate_config(config)


def test_compact_manifest_validation(tmp_path):
    config = resolve_config(Path("experiments/pretrained_id/config.yaml"), "4b")
    prompt = config["prompt"]
    rows = [
        {
            "id": "a",
            "prompt": "trajectory",
            "metadata": {
                "source_dataset": "gloom_exfiltration",
                "ground_truth": 0,
                "prompt_template_sha256": prompt["template_sha256"],
            },
        }
    ]
    inputs = tmp_path / "prompts.jsonl"
    inputs.write_text(json.dumps(rows[0]) + "\n")
    digest = hashlib.sha256(inputs.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "output": str(inputs),
                "output_sha256": digest,
                "rows": 1,
                "prompt_set_id": prompt["prompt_set_id"],
                "student_template_sha256": prompt["template_sha256"],
            }
        )
    )
    config["scope"].update(
        input=str(inputs),
        input_sha256=digest,
        manifest=str(manifest),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        rows=1,
        sources=["gloom_exfiltration"],
    )
    assert validate_inputs(config) == rows
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest checksum"):
        validate_inputs(config)
