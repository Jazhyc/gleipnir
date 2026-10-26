import json

from experiments.probe_sleightbench import score


def candidate(architecture: str, epoch: int, objective: str) -> dict:
    return {
        "architecture": architecture,
        "epoch": epoch,
        "probes": [f"{architecture}-{objective}-s{s}" for s in range(5)],
        "id_seed_median": {"pauroc_at_20": 0.8, "auroc": 0.9, "neg_brier": -0.1},
    }


def test_picks_merge_selected_and_family_roles(tmp_path, monkeypatch):
    selection = {
        objective: {
            "selected": candidate("attention", 8, objective),
            "families": {
                "attention": candidate("attention", 8, objective),
                "multimax": candidate("multimax", 16, objective),
            },
        }
        for objective in ("hard", "soft")
    }
    (tmp_path / "selection_ntp.json").write_text(json.dumps(selection))
    monkeypatch.setattr(score, "PROBES", tmp_path)
    picks = score.picks()
    assert [(p["objective"], p["architecture"], p["epoch"]) for p in picks] == [
        ("hard", "attention", 8),
        ("hard", "multimax", 16),
        ("soft", "attention", 8),
        ("soft", "multimax", 16),
    ]
    assert picks[0]["roles"] == ["selected", "family:attention"]
    assert picks[1]["roles"] == ["family:multimax"]
