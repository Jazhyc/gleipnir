import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "experiments"
    / "tool_trajectory_monitoring"
    / "kimi_k26_ood_benchmark.json"
)


def test_frozen_kimi_k26_campaign_identity_and_cost() -> None:
    config = json.loads(CONFIG_PATH.read_text())

    assert config["status"] in {
        "frozen_evaluation",
        "quality_canary_passed_full_evaluation_pending",
        "completed",
    }
    assert config["scope"]["rows"] == 6_395
    assert config["model"]["openrouter_id"] == "moonshotai/kimi-k2.6"
    assert config["request"]["provider_only"] == "Inceptron"
    assert config["request"]["allow_fallbacks"] is False
    assert config["request"]["reasoning_effort"] == "none"
    assert config["request"]["top_logprobs"] >= 2
    assert config["quality_canary"]["rows"] == 600
    assert "pAUROC@20" in config["quality_canary"]["reported_metric"]
    if "canary_outcome" in config:
        assert config["canary_outcome"]["rows"] == 600
        assert config["canary_outcome"]["provider"] == "Inceptron"
        assert config["canary_outcome"]["reasoning_tokens"] == 0
    assert config["prompt"]["template_sha256"] == (
        "2418cc55801deead8983d3bde6e35b59603f0080711ce8e12e1d7299487ac128"
    )

    tokens = config["token_audit"]
    request = config["request"]
    projected = (
        tokens["input_tokens"]
        * request["provider_max_prompt_price_per_million"]
        + config["scope"]["rows"]
        * tokens["expected_completion_tokens_per_row"]
        * request["provider_max_completion_price_per_million"]
    ) / 1_000_000
    assert projected == pytest.approx(request["projected_uncached_full_cost_usd"])
    assert projected < request["maximum_campaign_cost_usd"]


def test_frozen_inputs_and_balanced_canary_prefix() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    full_path = ROOT / config["scope"]["input"]
    canary_path = ROOT / config["canary"]["input"]

    assert hashlib.sha256(full_path.read_bytes()).hexdigest() == (
        config["scope"]["input_sha256"]
    )
    assert hashlib.sha256(canary_path.read_bytes()).hexdigest() == (
        config["canary"]["input_sha256"]
    )

    rows = [json.loads(line) for line in canary_path.read_text().splitlines()[:12]]
    strata = {
        (row["metadata"]["source_dataset"], row["metadata"]["ground_truth"])
        for row in rows
    }
    assert len(rows) == 12
    assert len(strata) == 12
