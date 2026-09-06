import hashlib
import json

import pytest

from experiments.monitoring_prefix_supervision.prepare_training import prepare_training


def test_prepare_training_requires_complete_cache_and_preserves_parents(
    tmp_path, monkeypatch
):
    # Numeric/64-row audit validation has separate tests; keep this fixture tiny.
    monkeypatch.setattr(
        "experiments.monitoring_prefix_supervision.prepare_training.validate_fresh_audit",
        lambda *args: None,
    )
    source = tmp_path / "parents.jsonl"
    parents = [
        {
            "prompt_id": p,
            "student_target": "Kimi",
            "label": 1,
            "student_prompt": "rubric\n<agent_trajectory>\na\nb\n</agent_trajectory>",
        }
        for p in ("p", "q")
    ]
    source.write_text("".join(json.dumps(r) + "\n" for r in parents))
    reference = {
        "id": "p:2",
        "parent_prompt_id": "p",
        "end_character": 2,
        "prefix_sha256": hashlib.sha256(b"a\n").hexdigest(),
        "rendered_user_prompt_sha256": "teacher-prompt",
    }
    refs = tmp_path / "refs.jsonl"
    refs.write_text(json.dumps(reference) + "\n")
    contract = {
        "manifest": {
            "source": str(source),
            "rows": 1,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "references_sha256": hashlib.sha256(refs.read_bytes()).hexdigest(),
        }
    }
    contract_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "contract.json").write_text(json.dumps(contract))
    record = {
        **reference,
        "contract_sha256": contract_hash,
        "score": 0.5,
        "logprob_0": -1.0,
        "logprob_1": -1.0,
    }
    (cache / "logits.jsonl").write_text(json.dumps(record) + "\n")
    output = tmp_path / "paired.jsonl"
    with pytest.raises(FileNotFoundError):
        prepare_training(cache, refs, output)
    assert not output.exists()
    (cache / "complete.json").write_text(
        json.dumps({"rows": 1, "contract_sha256": contract_hash})
    )
    with pytest.raises(FileNotFoundError):
        prepare_training(cache, refs, output)
    (cache / "fresh_audit.json").write_text("{}")
    result = prepare_training(cache, refs, output)
    assert result["parent_rows"] == 2
    assert result["parents_with_prefix"] == 1
    paired = [json.loads(line) for line in output.read_text().splitlines()]
    for original, augmented in zip(parents, paired, strict=True):
        assert all(augmented[k] == v for k, v in original.items())
    assert paired[0]["prefix_soft_target"] == 0.5
    assert "prefix_soft_target" not in paired[1]
    with pytest.raises(FileExistsError):
        prepare_training(cache, refs, output)
    (cache / "logits.jsonl").write_text("")
    with pytest.raises(ValueError, match="complete"):
        prepare_training(cache, refs, tmp_path / "incomplete.jsonl")
