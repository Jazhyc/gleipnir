import json

import pytest

from gleipnir.prefix_cache import binary_score, validate_resume


def test_binary_score_stable():
    assert binary_score(-10000, -10000) == 0.5
    assert binary_score(-10000, 0) == 1
    with pytest.raises(ValueError):
        binary_score(float("nan"), 0)


def test_resume_checks_contract_and_duplicate(tmp_path):
    path = tmp_path / "cache.jsonl"
    row = dict(
        id="a",
        prefix_sha256="p",
        rendered_user_prompt_sha256="u",
        contract_sha256="c",
        logprob_0=-1,
        logprob_1=-1,
        score=0.5,
    )
    refs = {"a": row}
    assert validate_resume(path, refs, "c") == set()
    path.write_text(json.dumps(row) + "\n")
    assert validate_resume(path, refs, "c") == {"a"}
    with pytest.raises(ValueError, match="contract"):
        validate_resume(path, refs, "wrong")
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        validate_resume(path, refs, "c")


def test_resume_rejects_corrupt_score(tmp_path):
    path = tmp_path / "cache.jsonl"
    row = dict(
        id="a",
        prefix_sha256="p",
        rendered_user_prompt_sha256="u",
        contract_sha256="c",
        logprob_0=-1,
        logprob_1=-1,
        score=0.7,
    )
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="probability"):
        validate_resume(path, {"a": row}, "c")
