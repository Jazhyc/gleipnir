import hashlib
import math

import pytest

from gleipnir.branch_data import BranchDataset


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + "\nassistant\n"

    def encode(self, text, **kwargs):
        return list(text.encode())


def dataset():
    d = BranchDataset.__new__(BranchDataset)
    d.tokenizer = Tokenizer()
    d.parents = [
        {
            "prompt_id": 0,
            "dataset": "a",
            "index": "x",
            "teacher_rendered_prompt_sha256": "teacher",
            "student_prompt": (
                "rubric\n<agent_trajectory>\nabc def ghi\n</agent_trajectory>\n"
            ),
        }
    ]
    d.soft = {("a", "x"): {"rendered_prompt_sha256": "teacher", "soft_target": 0.9}}
    d.prefixes = {
        0: [
            {
                "id": str(end),
                "parent_prompt_id": 0,
                "end_character": end,
                "prefix_sha256": hashlib.sha256(b"abc def ghi\n"[:end]).hexdigest(),
                "logprob_0": 0.0,
                "logprob_1": -1.0,
                "score": 1 / (1 + math.e),
                "contract_sha256": "contract",
            }
            for end in (3, 7)
        ]
    }
    return d


def test_all_boundaries_and_unchanged_full_target():
    d = dataset()
    r = d[0]
    assert len(r["plan"].requests) == 3
    assert r["targets"] == [1 / (1 + math.e), 1 / (1 + math.e), 0.9]
    assert b"ghi" not in bytes(r["plan"].requests[0])
    assert b"abc def ghi" in bytes(r["plan"].requests[-1])
    d.prefixes[0] = []
    assert d[0]["targets"] == [0.9]


def test_full_teacher_prompt_mismatch_rejected():
    d = dataset()
    d.soft[("a", "x")]["rendered_prompt_sha256"] = "changed"
    with pytest.raises(ValueError, match="Kimi"):
        d[0]
