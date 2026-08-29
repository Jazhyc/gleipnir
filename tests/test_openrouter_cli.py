import json

import pytest

from gleipnir.openrouter_cli import (
    automatic_session_id,
    cache_token_totals,
    load_cache,
    load_prompts,
    shared_prompt_prefix,
)


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


def test_cache_rejects_request_settings_drift(tmp_path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "one", "prompt": "Current"}) + "\n")
    record = load_prompts(prompts)[0]
    cache = tmp_path / "cache.jsonl"
    cache.write_text(
        json.dumps(
            {
                "id": "one",
                "prompt_sha256": record.prompt_sha256,
                "request_settings_sha256": "old-settings",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="request settings mismatch"):
        load_cache(cache, {record.record_id: record}, "new-settings")


def test_shared_prefix_drives_stable_cache_session(tmp_path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps({"id": "one", "prompt": "Shared instructions\nFirst"})
        + "\n"
        + json.dumps({"id": "two", "prompt": "Shared instructions\nSecond"})
        + "\n"
    )
    records = load_prompts(prompts)
    prefix = shared_prompt_prefix(records)
    assert prefix == "Shared instructions\n"
    assert automatic_session_id("moonshotai/kimi-k3", prefix) == (
        automatic_session_id("moonshotai/kimi-k3", prefix)
    )


def test_cache_token_totals_handles_missing_telemetry() -> None:
    assert cache_token_totals(
        {
            "one": {
                "usage": {"prompt_tokens": 120},
                "cache_usage": {"cached_tokens": 100, "cache_write_tokens": 20},
            },
            "two": {
                "usage": {"prompt_tokens": 100},
                "cache_usage": {"cached_tokens": 80},
            },
            "legacy": {},
        }
    ) == (220, 180, 20)
