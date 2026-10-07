from __future__ import annotations

import pytest

from experiments.tool_trajectory_monitoring.prepare_teacher_ood_cache import (
    FIREWORKS_COMPLETION_TOKENS_PER_ROW,
    FIREWORKS_INPUT_USD_PER_MILLION,
    FIREWORKS_OUTPUT_USD_PER_MILLION,
    OOD_SOURCES,
    cost_projection,
    expected_counts,
    validate_rows,
)
from experiments.tool_trajectory_monitoring.prepare_teacher_ood_canary import (
    select_balanced_ood_rows,
)


def representative_rows() -> list[dict[str, object]]:
    rows = []
    for source in OOD_SOURCES:
        source_row_index = 0
        for label, count in source.labels.items():
            for _ in range(count):
                rows.append({
                    "source": source.source,
                    "ground_truth": label,
                    "source_row_index": source_row_index,
                    "trajectory": "[USER] authorized task\n[ASSISTANT] completed it",
                })
                source_row_index += 1
    return rows


def test_ood_contract_has_paper_aggregate_counts() -> None:
    counts = expected_counts()
    assert sum(counts.values()) == 6_395
    assert (
        sum(count for (_, label), count in counts.items() if label == 0) == 3_189
    )
    assert (
        sum(count for (_, label), count in counts.items() if label == 1) == 3_206
    )
    assert counts[("rogue_eval", 1)] == 749
    assert counts[("shade_arena", 1)] == 418


def test_validate_rows_rejects_count_and_identity_drift() -> None:
    rows = representative_rows()
    validate_rows(rows)
    with pytest.raises(ValueError, match="counts drifted"):
        validate_rows(rows[:-1])
    duplicate = [dict(row) for row in rows]
    duplicate[-1]["source_row_index"] = duplicate[-2]["source_row_index"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_rows(duplicate)


def test_fireworks_cost_projection_uses_frozen_rates() -> None:
    projection = cost_projection(2_000_000, 1_000)
    assert projection["completion_tokens"] == (
        1_000 * FIREWORKS_COMPLETION_TOKENS_PER_ROW
    )
    assert projection["input_usd"] == pytest.approx(
        2 * FIREWORKS_INPUT_USD_PER_MILLION
    )
    assert projection["output_usd"] == pytest.approx(
        FIREWORKS_COMPLETION_TOKENS_PER_ROW
        * 1_000
        * FIREWORKS_OUTPUT_USD_PER_MILLION
        / 1_000_000
    )
    assert projection["usd_per_1000_rows"] == pytest.approx(
        projection["total_usd"]
    )


def test_balanced_ood_canary_selection_is_deterministic_and_interleaved() -> None:
    rows = [
        {
            "id": f"{source.source}-{label}-{index}",
            "metadata": {
                "source_dataset": source.source,
                "ground_truth": label,
            },
        }
        for source in OOD_SOURCES
        for label in (0, 1)
        for index in range(3)
    ]
    first = select_balanced_ood_rows(rows, rows_per_stratum=2, seed=7)
    second = select_balanced_ood_rows(rows, rows_per_stratum=2, seed=7)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert [
        (
            row["metadata"]["source_dataset"],
            row["metadata"]["ground_truth"],
        )
        for row in first[:12]
    ] == [
        (source.source, label) for source in OOD_SOURCES for label in (0, 1)
    ]
    with pytest.raises(ValueError, match="need 4"):
        select_balanced_ood_rows(rows, rows_per_stratum=4, seed=7)
