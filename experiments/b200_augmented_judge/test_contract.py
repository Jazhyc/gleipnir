"""Protect the serving intervention and paired preference identities."""

import copy
import json

import pytest

from experiments.b200_augmented_judge.run import bind_predictions, judge_command


@pytest.fixture
def parent() -> list[str]:
    return [
        "python",
        "-m",
        "old.server",
        "--worker-cls",
        "old.Worker",
        "--model",
        "old-model",
        "--runner",
        "pooling",
        "--max-model-len",
        "32768",
        "--quantization",
        "frost_fp4",
        "--kv-cache-dtype",
        "bfloat16",
        "--hf-overrides",
        json.dumps(
            {
                "classifier_from_token": ["0", "1"],
                "text_config": {
                    "classifier_from_token": ["0", "1"],
                    "hidden_size": 2560,
                },
            }
        ),
        "--additional-config",
        json.dumps(
            {
                "monitor_score": {"token_ids": [15, 16]},
                "runtime_migration": {"vllm": "0.31.0"},
                "serving_condition": {"attention_precision": "mxfp8"},
                "gleipnir_frost_fp4": {"kernel.py": "hash"},
            }
        ),
    ]


@pytest.mark.parametrize("precision", ["bf16", "optimized"])
def test_ab_layout_preserves_context_and_parent(parent, precision):
    original = copy.deepcopy(parent)
    command = judge_command(parent, precision, {"new.py": "new-hash"})
    assert parent == original
    overrides = json.loads(command[command.index("--hf-overrides") + 1])
    assert overrides["classifier_from_token"] == ["A", "B"]
    assert overrides["text_config"]["classifier_from_token"] == ["A", "B"]
    assert overrides["text_config"]["hidden_size"] == 2560
    additional = json.loads(command[command.index("--additional-config") + 1])
    assert additional["monitor_score"]["token_ids"] == [32, 33]
    assert command[command.index("--max-model-len") + 1] == "32768"
    assert command[command.index("--runner") + 1] == "pooling"
    if precision == "optimized":
        assert additional["serving_condition"]["attention_precision"] == "mxfp8"
        assert additional["gleipnir_frost_fp4"] == {"kernel.py": "hash"}
        assert command[command.index("--quantization") + 1] == "frost_fp4"
    else:
        assert "--quantization" not in command
        assert additional["serving_condition"]["quantization"] is None
        assert command[command.index("--kv-cache-dtype") + 1] == "auto"


def test_paired_metadata_survives_binding():
    rows = [
        {
            "id": "pair-A",
            "student_prompt": "private input",
            "label": 0,
            "condition": "preferred_injected",
            "pair_id": "pair",
            "order": 0,
        }
    ]
    result = bind_predictions([{"id": "pair-A", "score": 0.75}], rows)[0]
    assert result["label"] == 0
    assert result["pair_id"] == "pair"
    assert result["condition"] == "preferred_injected"
    assert result["p_B"] == 0.75
    assert "student_prompt" not in result


@pytest.mark.parametrize("ids", [["A"], ["B", "A"], ["A", "A"]])
def test_incomplete_reordered_duplicate_scores_fail(ids):
    with pytest.raises(ValueError):
        bind_predictions(
            [{"id": i, "score": 0.5} for i in ids],
            [{"id": "A", "label": 0}, {"id": "B", "label": 1}],
        )
