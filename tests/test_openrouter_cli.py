import json

import pytest

from gleipnir.openrouter_cli import load_cache, load_prompts


def test_load_prompts_requires_unique_ids(tmp_path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps({"id": "one", "prompt": "First"})
        + "\n"
        + json.dumps({"id": "one", "prompt": "Second"})
        + "\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_prompts(path)


def test_cache_rejects_prompt_drift(tmp_path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "one", "prompt": "Current"}) + "\n")
    record = load_prompts(prompts)[0]
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps({"id": "one", "prompt_sha256": "stale"}) + "\n")
    with pytest.raises(ValueError, match="prompt mismatch"):
        load_cache(cache, {record.record_id: record})
