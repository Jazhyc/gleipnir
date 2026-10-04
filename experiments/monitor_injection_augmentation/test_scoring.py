"""Verify cache resumption, explicit logprobs and failure on input/label drift."""

import math
from types import SimpleNamespace

import pytest

from gleipnir.monitoring_scoring import completed_predictions, predictions_for


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + "\nAssistant:"


class Engine:
    def __init__(self):
        self.calls = 0

    def generate(self, inputs, sampling, **kwargs):
        self.calls += 1
        return [
            SimpleNamespace(
                prompt_token_ids=r["prompt_token_ids"],
                outputs=[
                    SimpleNamespace(
                        token_ids=[49],
                        logprobs=[
                            {
                                48: math.log(0.25),
                                49: math.log(0.75),
                            }
                        ],
                    )
                ],
            )
            for r in inputs
        ]


def test_resumption_preserves_raw_decision_logprobs_and_campaign_status(tmp_path):
    engine = Engine()
    rows = [
        {
            "id": "one",
            "prompt": "Review transcript.",
            "metadata": {
                "ground_truth": 0,
                "source_dataset": "source",
            },
        }
    ]
    path = tmp_path / "scores.jsonl"
    ident = {"campaign": "test"}
    settings = {"batch_rows": 128, "max_model_len": 32768}

    def score(records):
        return predictions_for(
            engine,
            None,
            None,
            Tokenizer(),
            records,
            "01",
            path,
            ident,
            "id",
            engine=settings,
            status_dir=tmp_path,
        )

    result = score(rows)
    assert engine.calls == 1 and result[0]["score"] == pytest.approx(0.75)
    assert set(result[0]["raw_decision_logprobs"]) == {"48", "49"}
    assert score(rows) == result and engine.calls == 1
    assert completed_predictions(path, rows, ident) == result
    assert (tmp_path / "status.json").exists()
    with pytest.raises(ValueError, match="identity drift"):
        score([{**rows[0], "prompt": "Changed transcript."}])
    with pytest.raises(ValueError, match="identity drift"):
        score([{**rows[0], "metadata": {**rows[0]["metadata"], "ground_truth": 1}}])
