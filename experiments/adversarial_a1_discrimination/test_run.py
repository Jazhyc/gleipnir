from .run import comparison


def test_all_alarm_does_not_imply_ranking_collapse():
    result = comparison([0.7, 0.75], [0.8, 0.9], 0.2)
    assert result["honest_flagged"] == result["harmful_flagged"] == 2
    assert result["auroc"] == result["paired_rank_fraction"] == 1
    tied = comparison([0.8, 0.8], [0.8, 0.8], 0.2)
    assert tied["auroc"] == tied["paired_rank_fraction"] == 0.5
    assert tied["tied"] == 2
