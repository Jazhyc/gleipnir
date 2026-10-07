from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.tool_trajectory_monitoring import (
    benchmark_qwen_binary_reasoning_ood as benchmark,
)
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import load_json


def fake_output(
    text: str,
    token_ids: list[int],
    *,
    prompt_token_ids: list[int],
    finish_reason: str = "stop",
    stop_reason: str | None = None,
    logprobs: list[dict[int, SimpleNamespace]] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_ids=prompt_token_ids,
        outputs=[
            SimpleNamespace(
                text=text,
                token_ids=token_ids,
                finish_reason=finish_reason,
                stop_reason=stop_reason,
                logprobs=logprobs,
            )
        ],
    )


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text == "\nPrediction:":
            return [31, 32]
        if text == "Concrete unauthorized action.\nPrediction:1":
            return [21, 22, 23, 31, 32, 101]
        raise AssertionError(f"unexpected text to tokenize: {text!r}")


def test_reasoned_prediction_retains_rationale_and_binary_logprobs() -> None:
    record = {
        "id": "row-1",
        "metadata": {
            "source_dataset": "source",
            "ground_truth": 1,
            "rendered_prompt_sha256": "source-hash",
        },
    }
    generation_prompt = "rendered prompt"
    continuation = "Concrete unauthorized action.\nPrediction:"
    rationale_output = fake_output(
        continuation,
        [21, 22, 23],
        prompt_token_ids=[1, 2, 3],
        stop_reason="\nPrediction:",
    )
    margin_output = fake_output(
        "1",
        [101],
        prompt_token_ids=[1, 2, 3, 21, 22, 23, 31, 32],
        logprobs=[
            {
                100: SimpleNamespace(logprob=-2.0),
                101: SimpleNamespace(logprob=-0.1),
            }
        ],
    )
    row = benchmark.reasoned_prediction_row(
        record,
        generation_prompt,
        rationale_output,
        generation_prompt + continuation,
        margin_output,
        FakeTokenizer(),
        [100, 101],
        "\nPrediction:",
        "config-hash",
        32_768,
    )
    assert row["rationale"] == "Concrete unauthorized action."
    assert row["prediction"] == 1
    assert row["score"] > 0.5
    assert row["logprob_0"] == -2.0
    assert row["logprob_1"] == -0.1
    assert row["rationale_tokens"] == 3
    assert row["output_tokens"] == 6
    assert row["prompt_tokens"] == 3
    assert row["margin_prompt_tokens"] == 8
    assert row["served_output_tokens"] == 4
    assert row["completion_retokenization_delta"] == 2


def paired_rows(scores: list[float]) -> list[dict]:
    rows = []
    for source in ("a", "b"):
        for index, (label, score) in enumerate(zip((0, 0, 1, 1), scores, strict=True)):
            rows.append(
                {
                    "id": f"{source}-{index}",
                    "source": source,
                    "label": label,
                    "score": score,
                    "prediction": int(score >= 0.5),
                    "prompt_tokens": 10,
                    "output_tokens": 12,
                    "served_output_tokens": 12,
                    "completion_retokenization_delta": 0,
                    "rationale_tokens": 10,
                    "reasoning_context_tokens": 10,
                    "synthetic_boundary_tokens": 0,
                    "margin_prompt_tokens": 20,
                    "rationale": "analysis",
                    "contains_think_close": False,
                    "rationale_finish_reason": "stop",
                    "rationale_truncated": False,
                    "boundary_recovery": "none",
                }
            )
    return rows


def test_reasoning_summary_and_direct_comparison() -> None:
    direct = paired_rows([0.1, 0.2, 0.8, 0.9])
    reasoned = paired_rows([0.05, 0.3, 0.7, 0.95])
    summary = benchmark.summarize_reasoning(reasoned)
    assert summary["metrics"]["macro"]["macro"]["auroc"] == 1.0
    assert summary["reasoning"]["tokens_total"] == 80
    assert summary["output_tokens"]["total"] == 96
    comparison = benchmark.compare_direct_baseline(reasoned, direct)
    assert comparison["delta_reasoned_minus_direct"]["macro"]["auroc"] == 0.0
    assert comparison["decision_agreement"] == 1.0
    assert comparison["mean_absolute_score_delta"] == pytest.approx(0.075)


def test_frozen_config_and_prompt_contract() -> None:
    config = load_json(
        Path(
            "experiments/tool_trajectory_monitoring/"
            "qwen_binary_reasoning_ood_benchmark.json"
        )
    )
    benchmark.validate_config(config)
    assert config["prompt"]["enable_thinking"] is False
    assert config["generation"]["temperature"] == 0.0
    assert config["generation"]["max_tokens"] == 4_096
    assert config["generation"]["stop"] == "\nPrediction:"
    assert config["scoring"]["tokens"] == ["0", "1"]
    assert config["engine"]["audited_max_prompt_tokens"] == 30_389
    assert config["engine"]["audited_total_prompt_tokens"] == 60_584_384
    assert config["metrics"][0].startswith("primary:")

    instruction = Path(config["prompt"]["instruction_file"]).read_text(
        encoding="utf-8"
    )
    assert "Write the concise analysis first" in instruction
    assert "Prediction:0" in instruction
    assert "Prediction:1" in instruction
    assert "Sinha" not in instruction


def test_frozen_config_rejects_insufficient_context() -> None:
    config = load_json(
        Path(
            "experiments/tool_trajectory_monitoring/"
            "qwen_binary_reasoning_ood_benchmark.json"
        )
    )
    config["engine"]["max_model_len"] = 34_000
    with pytest.raises(ValueError, match="does not cover"):
        benchmark.validate_config(config)
