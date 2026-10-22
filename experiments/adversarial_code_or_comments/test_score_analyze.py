"""Scoring records, the endpoint client, the margin model and prediction tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.adversarial_code_or_comments import common
from experiments.adversarial_code_or_comments.analyze import (
    CELLS,
    contrasts,
    fe_ols,
    kendall_tau,
    main_grid,
    margin_model,
)
from experiments.adversarial_code_or_comments.llm import Client, endpoint, priced
from experiments.adversarial_code_or_comments.score import (
    MockScorer,
    record,
    rows_for,
    with_instruction,
)
from gleipnir.adversarial.contract import join_prompt, split_prompt

CFG = common.config()
EFFECTS = {"code": 3.0, "claim": 1.5, "any_text": 0.8}


def cell_margin(code: str, text: str) -> float:
    m = EFFECTS["code"] * (code == "backdoored")
    if text != "none":
        m += EFFECTS["any_text"]
    if f"{code}|{text}" in ("honest|misleading", "backdoored|truthful"):
        m += EFFECTS["claim"]
    return m


def synthetic(n_tasks: int, split: str, seed: int, shift: float = 0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tasks):
        task_effect = rng.normal(0, 2)
        for cell in CELLS:
            code, text = cell.split("|")
            margin = cell_margin(code, text) + task_effect + rng.normal(0, 0.5) + shift
            rows.append(
                {
                    "id": f"{split}{t}:{code}:{text}",
                    "task_id": f"{split}{t}",
                    "split": split,
                    "set": "main",
                    "code": code,
                    "text": text,
                    "repetition": 1 if text != "none" else 0,
                    "added_chars": 0
                    if text == "none"
                    else 600 + int(rng.integers(-60, 60)),
                    "margin": float(margin),
                    "score": float(1 / (1 + np.exp(-margin))),
                    "status": "ok",
                    "monitor": "synthetic",
                    "instruction": "student",
                }
            )
    for t in range(150):
        margin = rng.normal(-1, 2)
        rows.append(
            {
                "id": f"calibration:{t}",
                "task_id": f"c{t}",
                "split": "calibration",
                "set": "calibration",
                "code": "honest",
                "text": "none",
                "margin": float(margin),
                "score": float(1 / (1 + np.exp(-margin))),
                "status": "ok",
            }
        )
    return rows


def test_record_carries_the_margin_and_metadata():
    row = {
        "id": "1:honest:truthful",
        "metadata": {
            "task_id": "1",
            "split": "development",
            "set": "main",
            "code": "honest",
        },
    }
    out = record(
        row, {"score": 0.3, "logit_0": 2.0, "logit_1": 1.5}, "4b-v03", "student"
    )
    assert (
        out["margin"] == -0.5 and out["task_id"] == "1" and out["monitor"] == "4b-v03"
    )


def test_mock_scorer_is_deterministic_and_adapter_sensitive():
    scorer = MockScorer()
    assert scorer.score("x") == scorer.score("x")
    assert scorer.score("x") != scorer.score("x", adapter=False)


def test_rows_for_keeps_calibration_with_any_split():
    variants = [
        {"id": "a", "metadata": {"set": "calibration", "split": "calibration"}},
        {"id": "b", "metadata": {"set": "main", "split": "development"}},
        {"id": "c", "metadata": {"set": "main", "split": "heldout"}},
    ]
    assert [
        r["id"] for r in rows_for(variants, {"calibration", "main"}, "heldout")
    ] == [
        "a",
        "c",
    ]


def test_teacher_instruction_keeps_the_trajectory():
    user = join_prompt(common.instruction("student", CFG), "[user]\nq\n[assistant]")
    teacher = with_instruction(user, common.instruction("teacher", CFG))
    assert split_prompt(teacher).trajectory == split_prompt(user).trajectory
    assert split_prompt(teacher).head == common.instruction("teacher", CFG)


def test_endpoint_body_has_no_openrouter_fields_and_prices_tokens():
    config = endpoint(CFG["models"]["checker"])
    body = config.payload([{"role": "user", "content": "x"}])
    assert body["model"] == "Qwen/Qwen3.8-27B" and body["reasoning_effort"] is False
    assert "provider" not in body and "reasoning" not in body and "usage" not in body
    data = priced(
        {"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}, config
    )
    assert data["usage"]["cost"] == pytest.approx(0.6975)


def test_dry_run_client_needs_no_key_and_spends_nothing(monkeypatch):
    def no_key(*args, **kwargs):
        raise AssertionError("a dry run must not read keys")

    monkeypatch.setattr("experiments.adversarial_code_or_comments.llm.load_key", no_key)
    client = Client(CFG["models"]["writer"], 1.0, CFG, responder=lambda m: "hello")
    result = client.chat([{"role": "user", "content": "hi"}], attempt=2)
    assert result["text"] == "hello" and client.budget.spent_usd == 0.0


def test_paid_stages_refuse_a_draft_protocol():
    assert CFG["protocol_status"] == "draft"
    with pytest.raises(SystemExit, match="frozen"):
        common.require_frozen(CFG)


def test_fixed_effects_regression_recovers_effects():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(300), 4)
    x = rng.normal(size=(len(groups), 2))
    y = 2.0 * x[:, 0] - 1.0 * x[:, 1] + rng.normal(0, 3, 300)[groups]
    y += rng.normal(0, 0.1, len(groups))
    coef, se = fe_ols(x, y, [str(g) for g in groups])
    assert coef == pytest.approx([2.0, -1.0], abs=0.02)
    assert np.all(se > 0) and np.all(se < 0.05)


def test_contrasts_and_model_recover_the_planted_effects():
    rows = synthetic(300, "development", 1)
    grid = main_grid(rows, "development")
    per_task = contrasts(grid)
    for name, value in EFFECTS.items():
        assert np.mean(list(per_task[name].values())) == pytest.approx(value, abs=0.1)
    terms = margin_model(grid, CFG)["terms"]
    assert terms["backdoored"]["coef"] == pytest.approx(EFFECTS["code"], abs=0.15)
    assert terms["claims_special"]["coef"] == pytest.approx(EFFECTS["claim"], abs=0.15)


def test_kendall_tau():
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0


def test_predictions_hold_on_a_matching_heldout_and_fail_on_a_shift():
    pytest.importorskip("sklearn")  # thresholds come from gleipnir's metrics
    from experiments.adversarial_code_or_comments.analyze import compare, predict

    dev = synthetic(200, "development", 2)
    predictions = predict(dev, CFG)
    json.dumps(predictions)  # must be serialisable to commit
    assert predictions["larger_effect"] == "code"
    assert predictions["claim_at_least_half_code"] is True
    calibration = [r for r in dev if r["set"] == "calibration"]
    same = compare(predictions, synthetic(200, "heldout", 3) + calibration, CFG)
    assert same["same_monitor_success"] and same["second_monitor_success"]
    shifted = compare(predictions, synthetic(200, "heldout", 3, 2.0) + calibration, CFG)
    assert not shifted["same_monitor_success"]
    with pytest.raises(ValueError, match="held-out"):
        predict(dev + synthetic(5, "heldout", 4), CFG)
