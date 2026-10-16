from gleipnir.training.host_profile import interval_union


def test_interval_union_merges_overlaps_and_gaps() -> None:
    assert interval_union([(0, 2), (1, 3), (5, 6), (5.5, 5.7)]) == 4
    assert interval_union([]) == 0
