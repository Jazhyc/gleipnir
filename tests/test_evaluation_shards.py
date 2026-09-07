import pytest

from gleipnir.evaluation_shards import merge_predictions, partition_pending


def test_fresh_ood_shards_cover_every_row_once():
    rows = [{"id": str(i)} for i in range(6395)]
    parts = [partition_pending(rows, [], "frozen", 2, i) for i in range(2)]
    assert [len(part) for part in parts] == [3200, 3195]
    scored = [
        [{**row, "score": 0.5, "config_sha256": "frozen"} for row in part]
        for part in parts
    ]
    merged = merge_predictions(rows, scored, "frozen")
    assert [r["id"] for r in merged] == [r["id"] for r in rows]
    scored[0][0]["score"] = float("nan")
    with pytest.raises(ValueError, match="score drift"):
        merge_predictions(rows, scored, "frozen")


def test_partition_and_merge():
    rows = [{"id": str(i)} for i in range(15)]

    def scored(items):
        return [{**r, "score": 0.5, "config_sha256": "hash"} for r in items]

    saved = scored(rows[:3])
    left = partition_pending(rows, saved, "hash", 2, 0, 4)
    right = partition_pending(rows, saved, "hash", 2, 1, 4)
    assert [r["id"] for r in left] == list(map(str, [3, 4, 5, 6, 11, 12, 13, 14]))
    assert [r["id"] for r in right] == list(map(str, [7, 8, 9, 10]))
    parts = [saved, scored(left), scored(right)]
    assert merge_predictions(rows, parts, "hash") == scored(rows)
    with pytest.raises(ValueError, match="identity"):
        merge_predictions(rows, [*parts, saved], "hash")
    with pytest.raises(ValueError, match="incomplete"):
        merge_predictions(rows, parts[:2], "hash")
    with pytest.raises(ValueError, match="provenance"):
        partition_pending(rows, saved, "wrong", 2, 0)
