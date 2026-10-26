from __future__ import annotations

import numpy as np
import pytest

from experiments.adversarial_lab_liars_probe import probe


def meta(i: str, seed: int, worker: str, over: bool = False, posts: int = 1) -> dict:
    trajectory = "\n".join(["[TOOL CALLS]"] + ['board_post({"text": "x"})'] * posts)
    return {
        "id": i,
        "prompt": f"header mentions board_post here\n{trajectory}",
        "metadata": {"seed": seed, "worker": worker, "over_limit": over, "role": "x"},
    }


def pred(i: str, seed: int, worker: str, status: str = "ok", score=0.5) -> dict:
    return {
        "id": i,
        "seed": seed,
        "worker": worker,
        "status": status,
        "score": score,
        "activation_file": f"{i}.npy",
    }


def test_join_keeps_latest_ok_and_counts_posts():
    rows = probe.join_rows(
        [meta("a", 1, "B", posts=3), meta("b", 1, "C", over=True)],
        [pred("a", 1, "B", status="error", score=None), pred("a", 1, "B", score=0.2)],
    )
    assert [r["id"] for r in rows] == ["a"]
    assert rows[0]["score"] == 0.2 and rows[0]["posts"] == 3


@pytest.mark.parametrize(
    "metas,preds",
    [
        ([meta("a", 1, "B")], []),  # in-limit row unscored
        ([meta("a", 1, "B")], [pred("a", 1, "B"), pred("z", 1, "B")]),  # no metadata
        ([meta("a", 1, "B", over=True)], [pred("a", 1, "B")]),  # over limit, scored
        ([meta("a", 1, "B")], [pred("a", 2, "B")]),  # seed disagrees
        ([meta("a", 1, "B"), meta("a", 1, "B")], [pred("a", 1, "B")]),  # dup id
    ],
)
def test_join_rejects_bad_inputs(metas, preds):
    with pytest.raises(ValueError):
        probe.join_rows(metas, preds)


def test_seed_folds_one_fold_per_seed():
    seeds = [10703, 10701, 10701, 10702, 10703]
    folds = probe.seed_folds(seeds)
    assert folds == {10701: 0, 10702: 1, 10703: 2}
    assert len(set(folds.values())) == len(folds)


def synthetic(n_seeds=5, per_seed=12, seed=0):
    rng = np.random.default_rng(seed)
    seeds = np.repeat(np.arange(n_seeds) + 100, per_seed)
    y = np.tile(np.arange(per_seed) % 2, n_seeds)
    x = rng.normal(size=(len(y), 6)) + y[:, None]
    x[:, 0] = seeds  # column 0 records each row's seed
    return {"L1": x, "L2": x.copy()}, y, seeds


def test_nested_oof_never_trains_on_the_scored_seed(monkeypatch):
    arrays, y, seeds = synthetic()
    seen: list[set] = []
    real = probe.fit_linear

    def spy(x, yy, c):
        seen.append(set(x[:, 0].tolist()))
        return real(x, yy, c)

    monkeypatch.setattr(probe, "fit_linear", spy)
    train = np.ones(len(y), dtype=bool)
    train[::7] = False  # some rows scored but never trained on
    score = np.ones(len(y), dtype=bool)
    oof, sels = probe.nested_oof(
        arrays, y, seeds, train, score, ["L1", "L2"], [0.1, 1.0], workers=1
    )
    assert not np.isnan(oof).any()
    assert sorted(s["seed"] for s in sels) == sorted(set(seeds.tolist()))
    for s in sels:
        assert s["train_rows"] == int((train & (seeds != s["seed"])).sum())
    # Every fit (inner and outer) of the fold holding out seed s excludes s;
    # each outer fold's refit sees exactly the other seeds' training rows.
    all_seeds = set(seeds.tolist())
    outer_fits = [f for f in seen if len(f) == len(all_seeds) - 1]
    assert {frozenset(all_seeds - f) for f in outer_fits} == {
        frozenset([s]) for s in all_seeds
    }


def test_nested_oof_scores_only_requested_rows():
    arrays, y, seeds = synthetic()
    score = seeds == 101
    oof, sels = probe.nested_oof(
        arrays, y, seeds, np.ones(len(y), bool), score, ["L1"], [1.0], workers=1
    )
    assert np.isnan(oof[~score]).all() and not np.isnan(oof[score]).any()
    assert [s["seed"] for s in sels] == [101]


def test_shuffle_within_seed_keeps_seed_counts():
    arrays, y, seeds = synthetic()
    mask = np.ones(len(y), dtype=bool)
    mask[:3] = False
    out = probe.shuffle_within_seed(y, seeds, mask, np.random.default_rng(0))
    assert (out[:3] == y[:3]).all()
    for s in set(seeds.tolist()):
        assert out[seeds == s].sum() == y[seeds == s].sum()


def test_calibration_rows_are_ordinary_workers_in_honest_episodes():
    rows = [
        {"role": "target", "honest_episode": True},
        {"role": "bystander", "honest_episode": True},
        {"role": "recruiter", "honest_episode": True},
        {"role": "bystander", "honest_episode": False},
    ]
    assert probe.calibration_mask(rows).tolist() == [True, True, False, False]


def test_line_flags_at_most_one_percent():
    scores = np.arange(208, dtype=float)
    line = probe.line_at(scores, 0.01)
    assert (scores > line).sum() == 2  # floor(0.01 * 208)
    ties = np.array([0.0] * 205 + [5.0] * 3)
    assert (ties > probe.line_at(ties, 0.01)).sum() == 0
    with pytest.raises(ValueError):
        probe.line_at(np.array([0.1, np.nan]), 0.01)
