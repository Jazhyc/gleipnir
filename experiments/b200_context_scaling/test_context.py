"""Window lineage and decoded length/identity guards."""

import copy

import pytest

from experiments.b200_context_scaling.run import bind_reference, build_windows, digest
from experiments.b200_context_scaling.summarize import aggregate


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, ids):
        return "".join(ids)


def source():
    text = "".join(chr(0x400 + i) for i in range(300))
    return [
        {
            "id": "parent",
            "prompt": text,
            "prompt_tokens": len(text),
            "prompt_sha256": digest(text),
            "label": 1,
        }
    ]


def test_windows_are_exact_unique_deterministic_and_do_not_inherit_labels():
    rows = build_windows(source(), CharacterTokenizer(), 32, 8)
    assert rows == build_windows(source(), CharacterTokenizer(), 32, 8)
    assert len({r["prompt_sha256"] for r in rows}) == 8
    assert all(len(r["prompt"]) == r["prompt_tokens"] == 32 for r in rows)
    assert all("label" not in r and r["source_id"] == "parent" for r in rows)


def test_source_drift_and_repeated_content_stop_construction():
    rows = source()
    rows[0]["prompt_tokens"] += 1
    with pytest.raises(ValueError, match="token count drift"):
        build_windows(rows, CharacterTokenizer(), 32, 2)
    rows = source()
    rows[0]["prompt"] = "x" * 300
    rows[0]["prompt_sha256"] = digest(rows[0]["prompt"])
    with pytest.raises(ValueError, match="unique exact-length"):
        build_windows(rows, CharacterTokenizer(), 32, 2)


def test_warm_reference_identity_cannot_change(monkeypatch):
    monkeypatch.setattr(
        "experiments.b200_context_scaling.run.os.kill", lambda pid, sig: None
    )
    server = {
        "status": "ready",
        "pid": 123,
        "command": ["python", "--runner", "pooling"],
        "config_sha256": "config",
        "score_sources": {"worker": "hash"},
    }
    prior = {"status": "complete", "server": copy.deepcopy(server)}
    bind_reference(server, prior)
    server["command"].append("--scheduler-cls")
    with pytest.raises(ValueError, match="identity drift: command"):
        bind_reference(server, prior)


def test_incomplete_sweep_cannot_be_exported():
    with pytest.raises(ValueError, match="incomplete repeat/workload"):
        aggregate([])
