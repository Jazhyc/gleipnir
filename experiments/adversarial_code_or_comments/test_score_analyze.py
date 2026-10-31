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
    phrase_matched_tasks,
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


def test_paid_stages_refuse_a_draft_protocol(monkeypatch):
    # The clean-tree check is separate; fake it so this tests the status rule only.
    monkeypatch.setattr(common, "git_clean", lambda *paths: True)
    with pytest.raises(SystemExit, match="frozen"):
        common.require_frozen({**CFG, "protocol_status": "draft"})
    common.require_frozen({**CFG, "protocol_status": "frozen"})


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


def test_phrase_matched_tasks_need_agreement_between_the_two_claims():
    def text(task, kind, body, attempt=1, status="ok"):
        return {
            "task_id": task,
            "kind": kind,
            "attempt": attempt,
            "status": status,
            "text": body,
        }

    p = "It Departs from what the problem requires here."
    texts = [
        text("both", "honest_misleading", p),
        text("both", "backdoor_truthful", p),
        text("neither", "honest_misleading", "wrong answer"),
        text("neither", "backdoor_truthful", "wrong answer"),
        text("split", "honest_misleading", p),
        text("split", "backdoor_truthful", "wrong answer"),
        # the final attempt decides; a failed final attempt drops the task
        text("late", "honest_misleading", p),
        text("late", "backdoor_truthful", p, attempt=1),
        text(
            "late",
            "backdoor_truthful",
            "too long",
            attempt=2,
            status="length_out_of_band",
        ),
    ]
    assert phrase_matched_tasks(texts) == {"both", "neither"}


class FakeScorer:
    def __init__(self, monitor, shift=0.0):
        self.monitor = monitor
        self.shift = shift
        self.spec = type("Spec", (), {"adapter": "a"})()

    def score(self, user_prompt, adapter=True):
        return {"score": 0.5 + self.shift if adapter else 0.1}


def canary_inputs():
    tasks = ["t1", "t2", "t3", "t4"]
    pairs = {
        t: {c: {"prompt": f"{c} {t}"} for c in ("honest", "backdoored")} for t in tasks
    }
    return pairs, {"development_pool": tasks}


@pytest.mark.parametrize("shift, passed", [(0.0, True), (0.05, False)])
def test_canary_compares_the_4b_student_run_to_its_reference(
    monkeypatch, shift, passed
):
    from experiments.adversarial_code_or_comments import score

    reference = {
        f"{p}:{t}": 0.5 for p in ("clean", "original") for t in ("t1", "t2", "t3", "t4")
    }
    monkeypatch.setattr(score, "reference_scores", lambda cfg: reference)
    pairs, selection = canary_inputs()
    check = score.canary(
        FakeScorer("4b-v03", shift), pairs, selection, CFG, None, False
    )
    assert check["compared_to_reference"] is True
    assert check["passed"] is passed


def test_canary_skips_the_reference_for_other_monitors(monkeypatch):
    from experiments.adversarial_code_or_comments import score

    monkeypatch.setattr(score, "reference_scores", lambda cfg: {})
    pairs, selection = canary_inputs()
    check = score.canary(FakeScorer("27b-merged"), pairs, selection, CFG, None, False)
    assert check["compared_to_reference"] is False and check["passed"] is True
