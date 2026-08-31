from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from experiments.tool_trajectory_monitoring import (
    generate_gleipnir4b_reasoning_sample as sample,
)
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import load_json


def records() -> list[dict]:
    prompt = load_prompt_set().teacher
    rows = []
    for index in range(10):
        trajectory = f"user: task {index}\nassistant: action {index}"
        rows.append(
            {
                "id": f"row-{index}",
                "prompt": prompt.render(trajectory),
                "metadata": {
                    "source_dataset": "source",
                    "ground_truth": index % 2,
                },
            }
        )
    return rows


def prediction_rows(condition: str) -> list[dict]:
    return [
        {
            "id": f"row-{index}",
            "rationale": f"{condition} rationale {index}",
            "prediction": index % 2,
        }
        for index in range(3)
    ]


def test_hash_sample_is_order_independent_and_label_blind() -> None:
    source = records()
    selected = sample.select_sample(source, rows=4, seed=17)
    reversed_rows = list(reversed(deepcopy(source)))
    for row in reversed_rows:
        row["metadata"]["ground_truth"] = 1 - row["metadata"]["ground_truth"]
    repeated = sample.select_sample(reversed_rows, rows=4, seed=17)
    assert [row["id"] for row in selected] == [row["id"] for row in repeated]
    assert sample.selected_ids_sha256(selected) == sample.selected_ids_sha256(repeated)


def test_blinded_pairs_omit_condition_and_label_but_keep_separate_key() -> None:
    source = records()[:3]
    base = prediction_rows("base")
    gleipnir = prediction_rows("gleipnir")
    prompt = load_prompt_set().teacher
    judge, key = sample.blinded_pair_rows(
        source,
        base,
        gleipnir,
        blind_seed=9,
        binary_cache_prefix=prompt.cache_prefix,
    )
    assert len(judge) == len(key) == 3
    assert all(
        "ground_truth" not in row and "condition" not in str(row) for row in judge
    )
    assert all("trajectory" in row and "candidate_a" in row for row in judge)
    assert all(
        {row["candidate_a_condition"], row["candidate_b_condition"]}
        == {"base", "gleipnir"}
        for row in key
    )


def test_paired_summary_counts_reasoning_and_decision_changes() -> None:
    base = prediction_rows("base")
    gleipnir = prediction_rows("gleipnir")
    gleipnir[0]["prediction"] = 1
    summary = sample.paired_summary(base, gleipnir)
    assert summary["changed_rationales"] == 3
    assert summary["decision_disagreements"] == 1


def test_frozen_gleipnir_reasoning_config_contract() -> None:
    config = load_json(
        Path(
            "experiments/tool_trajectory_monitoring/"
            "gleipnir4b_reasoning_sample.json"
        )
    )
    # Artifact hashes are validated separately at runtime; this is the static contract.
    sample.validate_config(config)
    assert config["scope"]["sample_rows"] == 64
    assert config["gleipnir"]["train_rows"] == 21_837
    assert config["gleipnir"]["soft_loss_weight"] == 1.0
    assert config["gleipnir"]["direct_loss_weight"] == 0.0
    assert config["prompt"]["enable_thinking"] is False
