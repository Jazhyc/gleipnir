"""Guard sample identity and coordinate-comparison semantics."""

import pytest

from experiments.b200_training_direction.prepare import select
from experiments.b200_training_direction.run import replacement_command


def test_selection_is_order_independent_and_proportional():
    rows = [
        {
            "dataset": "x",
            "index": str(i),
            "lineage_group": str(i),
            "raw_source": "large" if i < 80 else "small",
            "label": i % 2,
        }
        for i in range(100)
    ]
    a = select(rows, 20, "fixed:")
    assert a == select(rows[::-1], 20, "fixed:")
    assert len(a) == 20
    assert sum(r["raw_source"] == "small" for r in a) == 4
    assert sum(r["label"] == 1 for r in a) == 10


def test_shared_lineage_cannot_be_split():
    with pytest.raises(ValueError, match="lineage"):
        select(
            [
                {"dataset": "x", "index": str(i), "lineage_group": "shared"}
                for i in range(2)
            ],
            1,
            "fixed:",
        )


def test_replacement_preserves_precision_scheduler_and_eager_flags():
    import json

    command = [
        "python",
        "--model",
        "/merged",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--scheduler-cls",
        "corrected",
        "--additional-config",
        json.dumps(
            {
                "serving_condition": {
                    "merged_model": "/merged",
                    "quantization": None,
                    "attention_precision": "bf16",
                }
            }
        ),
    ]
    result = replacement_command({"command": command}, "/base")
    assert command[2] == "/merged"  # Parent receipt stays immutable.
    assert result[:2] == command[:2] and result[3:-1] == command[3:-1]
    settings = json.loads(result[-1])["serving_condition"]
    assert settings == {
        "merged_model": "/base",
        "quantization": None,
        "attention_precision": "bf16",
    }
