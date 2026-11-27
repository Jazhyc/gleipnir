import pytest

pytest.importorskip("httpx")

from experiments.teachers.filter_margin_ood import run  # noqa: E402


def _truth() -> dict:
    return {f"{s}{k}": (s, int(k >= 10)) for s in ("a", "b") for k in range(20)}


def test_agreement_passes_on_matching_scores(monkeypatch) -> None:
    truth = _truth()
    monkeypatch.setattr(run, "labels", lambda split: truth)
    scores = {i: 0.9 if label else 0.1 for i, (_, label) in truth.items()}
    result = run.agreement(scores, dict(scores), "ood")
    assert result["passed"] and result["score_mae"] == 0


def test_agreement_fails_on_shifted_scores(monkeypatch) -> None:
    truth = _truth()
    monkeypatch.setattr(run, "labels", lambda split: truth)
    scores = {i: 0.9 if label else 0.1 for i, (_, label) in truth.items()}
    shifted = {i: 1 - s for i, s in scores.items()}
    assert not run.agreement(shifted, scores, "ood")["passed"]


def test_agreement_rejects_different_rows(monkeypatch) -> None:
    truth = _truth()
    monkeypatch.setattr(run, "labels", lambda split: truth)
    scores = {i: 0.5 for i in truth}
    with pytest.raises(ValueError):
        run.agreement(scores, dict(list(scores.items())[1:]), "ood")
