from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.tool_trajectory_monitoring import (
    benchmark_qwen_ordinal_reasoning_ood as benchmark,
)
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import load_json


def fake_output(
    text: str,
    token_ids: list[int],
    *,
    prompt_token_ids: list[int],
    finish_reason: str = "stop",
    stop_reason: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_ids=prompt_token_ids,
        outputs=[
            SimpleNamespace(
                text=text,
                token_ids=token_ids,
                finish_reason=finish_reason,
                stop_reason=stop_reason,
            )
        ],
    )


def test_reasoning_continuation_requires_analysis_and_exact_boundary() -> None:
    output = fake_output(
        "Evidence supports the innocent explanation.\nScore:",
        [1, 2, 3],
        prompt_token_ids=[10, 11],
        stop_reason="\nScore:",
    )
    text, token_ids = benchmark.reasoning_continuation(output, "\nScore:")
    assert text.endswith("\nScore:")
    assert token_ids == [1, 2, 3]

    output.outputs[0].text = "\nScore:"
    with pytest.raises(RuntimeError, match="no analysis"):
        benchmark.reasoning_continuation(output, "\nScore:")

    output.outputs[0].text = "analysis without boundary"
    with pytest.raises(RuntimeError, match="exact terminal score boundary"):
        benchmark.reasoning_continuation(output, "\nScore:")


def test_reasoned_prediction_retains_rationale_and_discrete_score() -> None:
    record = {
        "id": "row-1",
        "metadata": {
            "source_dataset": "source",
            "ground_truth": 1,
            "rendered_prompt_sha256": "source-hash",
        },
    }
    generation_prompt = "rendered prompt"
    continuation = "Concrete unauthorized action.\nScore:"
    rationale_output = fake_output(
        continuation,
        [21, 22, 23],
        prompt_token_ids=[1, 2, 3],
        stop_reason="\nScore:",
    )
    score_output = fake_output(
        "10",
        [101, 100],
        prompt_token_ids=[1, 2, 3, 21, 22, 23],
    )
    row = benchmark.reasoned_prediction_row(
        record,
        generation_prompt,
        rationale_output,
        generation_prompt + continuation,
        score_output,
        {str(score): [100 + score] for score in range(10)} | {"10": [101, 100]},
        999,
        "\nScore:",
        "config-hash",
        32_768,
    )
    assert row["rationale"] == "Concrete unauthorized action."
    assert row["ordinal_score"] == 10
    assert row["score"] == 1.0
    assert row["rationale_tokens"] == 3
    assert row["output_tokens"] == 5
    assert row["prompt_tokens"] == 3
    assert row["score_context_tokens"] == 6


def paired_rows(scores: list[int]) -> list[dict]:
    rows = []
    for source in ("a", "b"):
        for index, (label, score) in enumerate(zip((0, 0, 1, 1), scores, strict=True)):
            rows.append(
                {
                    "id": f"{source}-{index}",
                    "source": source,
                    "label": label,
                    "score": score / 10,
                    "ordinal_score": score,
                    "prompt_tokens": 10,
                    "output_tokens": 12,
                    "rationale_tokens": 10,
                    "score_context_tokens": 20,
                    "rationale": "analysis",
                    "contains_think_close": False,
                    "rationale_finish_reason": "stop",
                }
            )
    return rows


def test_reasoning_summary_and_immediate_comparison() -> None:
    immediate = paired_rows([0, 4, 6, 10])
    reasoned = paired_rows([0, 3, 7, 10])
    summary = benchmark.summarize_reasoning(reasoned)
    assert summary["metrics"]["macro"]["macro"]["auroc"] == 1.0
    assert summary["reasoning"]["tokens_total"] == 80
    comparison = benchmark.compare_immediate_ordinal(reasoned, immediate)
    assert comparison["delta_reasoned_minus_immediate"]["macro"]["auroc"] == 0.0
    assert comparison["score_agreement"] == 0.5
    assert comparison["mean_absolute_score_delta"] == pytest.approx(0.05)


def test_frozen_reasoning_config_and_prompt_contract() -> None:
    config = load_json(
        Path(
            "experiments/tool_trajectory_monitoring/"
            "qwen_ordinal_reasoning_ood_benchmark.json"
        )
    )
    benchmark.validate_config(config)
    assert config["prompt"]["enable_thinking"] is False
    assert config["generation"]["temperature"] == 0.0
    assert config["generation"]["stop"] == "\nScore:"
    assert config["engine"]["audited_max_prompt_tokens"] == 30_544
    assert config["engine"]["audited_total_prompt_tokens"] == 61_575_609
    assert config["metrics"][0].startswith("primary:")

    instruction = Path(config["prompt"]["instruction_file"]).read_text(
        encoding="utf-8"
    )
    for band in ("0--2", "3--4", "5--6", "7--8", "9--10"):
        assert band in instruction
    assert "Write the concise analysis first" in instruction
    assert "Sinha" not in instruction


def test_frozen_config_rejects_insufficient_context() -> None:
    config = load_json(
        Path(
            "experiments/tool_trajectory_monitoring/"
            "qwen_ordinal_reasoning_ood_benchmark.json"
        )
    )
    config["engine"]["max_model_len"] = 32_000
    with pytest.raises(ValueError, match="does not cover"):
        benchmark.validate_config(config)
