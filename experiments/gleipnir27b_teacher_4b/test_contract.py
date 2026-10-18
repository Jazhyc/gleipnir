import math
from pathlib import Path

import pytest
import yaml

from experiments.gleipnir27b_teacher_4b.prepare import (
    ARMS,
    CONTROL,
    arm_config,
    gleipnir_targets,
)
from gleipnir.data.monitoring import digest

ROOT = Path(__file__).resolve().parents[2]


def control() -> dict:
    return yaml.safe_load((ROOT / "data" / CONTROL / "config.yaml").read_text())


def test_arms_change_only_identity_text_rows_and_inputs() -> None:
    base = control()
    inputs = {"training": {"path": "data/x/training.jsonl", "sha256": "0" * 64}}
    arm = arm_config(base, "teacher", "GPU-test", "selection-sha", inputs, 8239, 1)
    changed = {k for k in set(arm) | set(base) if arm.get(k) != base.get(k)}
    assert changed == {
        "campaign_id",
        "extra_sources",
        "gpu_uuid",
        "merged_model",
        "inputs",
        "hypothesis",
        "intervention",
        "selection",
        "teacher_filter",
        "training_rows",
        "expected_steps",
        "expected_training_tokens",
    }
    for key in ("seed", "profile", "learning_rate", "soft_loss_weight", "model"):
        assert arm[key] == base[key]
    assert arm["campaign_id"] == ARMS["teacher"]
    assert arm["expected_steps"] == math.ceil(8239 / 32)
    assert "filter" not in arm["inputs"]
    assert arm["inputs"]["training"] == inputs["training"]
    assert "teacher prompt" in arm["intervention"]


def target_fixture(prompt: str = "raw") -> tuple[list[dict], list[dict], dict]:
    kimi = [
        {
            "index": "a",
            "dataset": "tool_trajectory/cot_red_handed",
            "label": 1,
            "rendered_prompt_sha256": digest(prompt),
            "soft_target": 0.2,
        }
    ]
    extracted = [
        {
            "id": "a",
            "label": 1,
            "source": "crh",
            "margin": 2.0,
            "logprob_0": -2.13,
            "logprob_1": -0.13,
            "outside_01_mass": 1e-4,
            "rendered_prompt_sha256": digest(f"<{prompt}>"),
        }
    ]
    return kimi, extracted, {"a": prompt}


def test_targets_bind_by_prompt_and_replace_only_the_teacher() -> None:
    kimi, extracted, prompts = target_fixture()
    (row,) = gleipnir_targets(kimi, extracted, prompts, lambda p: f"<{p}>")
    assert abs(row["soft_target"] - 1 / (1 + math.exp(-2.0))) < 1e-12
    assert row["soft_target"] == row["target_probs"]["positive"]
    assert (row["dataset"], row["index"], row["label"]) == (
        "tool_trajectory/cot_red_handed",
        "a",
        1,
    )
    assert row["rendered_prompt_sha256"] == kimi[0]["rendered_prompt_sha256"]


def test_targets_reject_render_or_source_drift() -> None:
    kimi, extracted, prompts = target_fixture()
    with pytest.raises(ValueError, match="drift"):
        gleipnir_targets(kimi, extracted, prompts, lambda p: p)
    extracted[0]["source"] = "gloom"
    with pytest.raises(ValueError, match="drift"):
        gleipnir_targets(kimi, extracted, prompts, lambda p: f"<{p}>")
