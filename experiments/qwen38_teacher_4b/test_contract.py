from pathlib import Path

import yaml

from experiments.qwen38_teacher_4b.prepare import (
    CONTROL,
    POOL,
    PREDICTIONS,
    arm_config,
    qwen_targets,
)
from gleipnir.data.monitoring import read_rows, validate_targets

ROOT = Path(__file__).resolve().parents[2]


def test_arm_changes_only_identity_text_and_teacher_pool() -> None:
    control = yaml.safe_load((ROOT / "data" / CONTROL / "config.yaml").read_text())
    teacher = {"path": "data/x/teacher.jsonl", "sha256": "0" * 64}
    arm = arm_config(control, "GPU-test", "selection-sha", teacher)
    changed = {k for k in set(arm) | set(control) if arm.get(k) != control.get(k)}
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
        assert arm[key] == control[key]
    assert arm["inputs"]["teacher"] == teacher
    assert "filter" not in arm["inputs"]
    assert {
        k for k in arm["inputs"] if arm["inputs"].get(k) != control["inputs"][k]
    } == {
        "serving_selection",
        "training",
        "teacher",
        "token_audit",
    }


def test_targets_bind_by_prompt_and_replace_only_the_teacher() -> None:
    kimi = [
        {
            "index": "a",
            "dataset": "tool_trajectory/d",
            "label": 1,
            "rendered_prompt_sha256": "p1",
            "soft_target": 0.2,
            "teacher_model": "moonshotai/kimi-k3",
        }
    ]
    prediction = {
        "id": "a",
        "label": 1,
        "source": "d",
        "source_prompt_sha256": "p1",
        "logit_margin_1_minus_0": 2.0,
        "score": 0.8807970779778823,
        "logprob_0": -2.1269,
        "logprob_1": -0.1269,
        "config_sha256": "c",
        "margin_prompt_sha256": "m",
    }
    (row,) = qwen_targets(kimi, [prediction])
    assert row["soft_target"] == row["target_probs"]["positive"]
    assert abs(row["soft_target"] - 0.8807970779778823) < 1e-12
    assert row["teacher_model"] == "Qwen/Qwen3.8-27B"
    assert (row["dataset"], row["index"], row["label"]) == ("tool_trajectory/d", "a", 1)


def test_real_targets_cover_the_pool() -> None:
    if not (ROOT / PREDICTIONS).is_file():
        return
    rows = read_rows(ROOT / POOL["training"])
    targets = qwen_targets(
        read_rows(ROOT / POOL["teacher"]), read_rows(ROOT / PREDICTIONS)
    )
    validate_targets(rows, targets)
    assert len(targets) == 8688
