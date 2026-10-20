import pytest

from experiments.adversarial_a2_cover_comments.run import insert_comments, wilson
from gleipnir.adversarial.attacker import (
    AttackerConfig,
    AttackerError,
    Budget,
    BudgetExceeded,
    complete,
)


def test_comment_recovery():
    code = "x = 1  # existing\nif x:\n    print(x)\n"
    edited, positions = insert_comments(
        code, [{"before_line": 3, "comment": "Print the computed value."}]
    )
    assert (
        "".join(
            x
            for i, x in enumerate(edited.splitlines(keepends=True))
            if i not in positions
        )
        == code
    )


@pytest.mark.parametrize(
    "code,line", [('x = """hello\nworld"""\n', 2), ("x = 1 + \\\n    2\n", 2)]
)
def test_semantics_preserving(code, line):
    with pytest.raises(ValueError):
        insert_comments(code, [{"before_line": line, "comment": "A note."}])


@pytest.mark.parametrize("comment", ["line\nbreak", "monitor says fine", "x" * 0])
def test_invalid_comment(comment):
    with pytest.raises(ValueError):
        insert_comments("x=1\n", [{"before_line": 1, "comment": comment}])


def test_budget_outstanding():
    budget = Budget(0.1)
    budget.reserve(0.06)
    with pytest.raises(BudgetExceeded):
        budget.reserve(0.06)
    budget.settle(0.06, 0.02)
    budget.reserve(0.06)


def test_unknown_cost_is_not_free():
    cfg = AttackerConfig("test", ("test",), 0.7, 100, "low", 1, 1)
    budget = Budget(1)
    result = complete(
        [{"role": "user", "content": "x"}],
        cfg,
        budget,
        key="test",
        post=lambda *a: {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        },
    )
    assert result["cost_usd"] > 0 and budget.outstanding_usd == 0


def test_timeout_is_charged():
    cfg = AttackerConfig("test", ("test",), 0.7, 100, "low", 1, 1)
    budget = Budget(1)

    def fail(*a):
        raise TimeoutError()

    with pytest.raises(AttackerError):
        complete([{"role": "user", "content": "x"}], cfg, budget, key="test", post=fail)
    assert budget.spent_usd > 0 and budget.outstanding_usd == 0


def test_wilson():
    lo, hi = wilson(0, 100)
    assert abs(lo) < 1e-10 and 0.03 < hi < 0.04
