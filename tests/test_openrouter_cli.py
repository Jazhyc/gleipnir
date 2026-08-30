import hashlib
import json

import pytest

from gleipnir.openrouter_cli import (
    automatic_session_id,
    cache_token_totals,
    declared_or_shared_prompt_prefix,
    load_cache,
    load_prompts,
    parse_args,
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


def test_declared_cache_prefix_stops_before_common_variable_text(tmp_path) -> None:
    prefix = "Shared instructions\n<trajectory>\n"
    digest = hashlib.sha256(prefix.encode()).hexdigest()
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": record_id,
                    "prompt": prefix + trajectory,
                    "metadata": {
                        "cache_prefix_chars": len(prefix),
                        "cache_prefix_sha256": digest,
                    },
                }
            )
            for record_id, trajectory in (("one", "[USER] a"), ("two", "[USER] b"))
        )
        + "\n"
    )
    records = load_prompts(prompts)
    assert shared_prompt_prefix(records) == prefix + "[USER] "
    assert declared_or_shared_prompt_prefix(records) == prefix


def test_declared_cache_prefix_rejects_drift(tmp_path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps(
            {
                "id": "one",
                "prompt": "Prompt",
                "metadata": {
                    "cache_prefix_chars": 3,
                    "cache_prefix_sha256": "0" * 64,
                },
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="hash does not match"):
        declared_or_shared_prompt_prefix(load_prompts(prompts))


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


def test_scale_checkpoint_and_progress_intervals_are_configurable() -> None:
    args = parse_args(
        [
            "--input",
            "prompts.jsonl",
            "--output",
            "scores.jsonl",
            "--model",
            "moonshotai/kimi-k3",
            "--checkpoint-every",
            "16",
            "--progress-every",
            "25",
        ]
    )
    assert args.checkpoint_every == 16
    assert args.progress_every == 25
