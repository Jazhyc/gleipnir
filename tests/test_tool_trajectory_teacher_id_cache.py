from __future__ import annotations

import pytest

from experiments.tool_trajectory_monitoring.prepare_teacher_id_cache import (
    validate_id_rows,
)


def rows() -> list[dict]:
    return [
        {
            "id": f"{source}:{label}:{index}",
            "source": source,
            "label": label,
            "trajectory": "action",
        }
        for source, counts in {
            "test_stride": {0: 369, 1: 577},
            "gloom_exfiltration": {0: 1_035, 1: 1_031},
        }.items()
        for label, count in counts.items()
        for index in range(count)
    ]


def test_validate_id_rows_accepts_exact_paper_counts() -> None:
    values = rows()
    validate_id_rows(values)
    with pytest.raises(ValueError, match="counts drifted"):
        validate_id_rows(values[:-1])


def test_validate_id_rows_rejects_duplicates_and_empty_trajectories() -> None:
    values = rows()
    values[-1]["id"] = values[0]["id"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_id_rows(values)
    values = rows()
    values[-1]["trajectory"] = ""
    with pytest.raises(ValueError, match="empty"):
        validate_id_rows(values)
