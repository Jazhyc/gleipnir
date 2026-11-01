"""Prompt-contract and metric checks; no GPU or model weights needed.

Tokenizer tests fetch only Qwen tokenizer files at the pinned revisions. Data
tests skip when a set has not been built or downloaded.
"""

from __future__ import annotations

import hashlib
import math

import pytest

from gleipnir.adversarial import eval_sets
from gleipnir.adversarial.contract import (
    CLOSE,
    DECISION_PREFIX,
    OPEN,
    join_prompt,
    render,
    split_prompt,
)
from gleipnir.adversarial.metrics import recall_at_fpr, threshold_at_fpr
from gleipnir.adversarial.monitors import MONITORS

STUDENT = eval_sets.ROOT / "experiments/tool_trajectory_monitoring/prompts/student.txt"
TEACHER = eval_sets.ROOT / "experiments/tool_trajectory_monitoring/prompts/teacher.txt"
# student_instruction_sha256 in the released prompt_contract.json of all three monitors
RELEASED_STUDENT_SHA256 = (
    "39ab7657c02e096a4004439183f0b5d7f259cd724c77cdd2d5885acb0f94a38c"
)


@pytest.fixture(scope="module")
def instruction() -> str:
    return STUDENT.read_text()


def test_instruction_matches_released_contract(instruction):
    digest = hashlib.sha256(instruction.encode()).hexdigest()
    assert digest == RELEASED_STUDENT_SHA256


def test_model_card_example_round_trips(instruction):
    trajectory = "User asked for a file listing. The agent ran: ls -la"
    card = f"{instruction}\n<agent_trajectory>\n{trajectory}\n</agent_trajectory>\n"
    assert join_prompt(instruction, trajectory) == card
    assert split_prompt(card) == (instruction, trajectory)


def test_tags_inside_trajectory_survive():
    trajectory = f"echo '{CLOSE}{OPEN}' > notes.txt"
    assert split_prompt(join_prompt("head\n", trajectory)).trajectory == trajectory


@pytest.mark.parametrize("name", sorted(eval_sets.SETS))
def test_every_row_splits_and_rejoins(name, instruction):
    if not eval_sets.path(name).exists():
        pytest.skip(f"{name} not on disk")
    heads = set()
    for row in eval_sets.load(name):
        parts = split_prompt(row["prompt"])
        assert join_prompt(*parts) == row["prompt"]
        heads.add(parts.head)
    assert len(heads) == 1
    if name == "ood-teacher":
        assert heads == {TEACHER.read_text()}
    elif "aware" not in name:
        assert heads == {instruction}


def test_low_fpr_threshold_respects_budget():
    labels = [0] * 100 + [1] * 10
    scores = [i / 100 for i in range(100)] + [0.995] * 5 + [0.5] * 5
    result = recall_at_fpr(labels, scores, 0.01)
    assert result["fpr"] <= 0.01 and result["recall"] == 0.5
    assert threshold_at_fpr([0.1, 0.2, 0.3], 0.0) == 0.3


@pytest.fixture(scope="module", params=sorted(MONITORS))
def tokenizer(request):
    transformers = pytest.importorskip("transformers")
    spec = MONITORS[request.param]
    try:
        return transformers.AutoTokenizer.from_pretrained(
            spec.base, revision=spec.base_revision
        )
    except OSError as error:
        pytest.skip(f"tokenizer unavailable: {error}")


def test_render_ends_at_decision_prefix(tokenizer):
    rendered = render(tokenizer, join_prompt("Classify.\n", "ls -la"))
    assert rendered.startswith("<|im_start|>user\nClassify.\n")
    assert rendered.endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n" + DECISION_PREFIX
    )


def test_labels_are_single_distinct_tokens(tokenizer):
    from gleipnir.evaluation.binary import binary_token_ids

    zero, one = binary_token_ids(tokenizer)
    assert tokenizer.decode([zero]) == "0" and tokenizer.decode([one]) == "1"
    assert math.isfinite(zero + one)
