import json
from pathlib import Path

import pytest

from experiments.release_serving import fetch
from experiments.release_serving.targets import ROOT, TARGETS


@pytest.mark.parametrize("key", sorted(TARGETS))
def test_recorded_command_serves_the_recorded_merge(key: str) -> None:
    target = TARGETS[key]
    command = json.loads((ROOT / target.server_receipt).read_text())["command"]
    assert command[command.index("--model") + 1] == str(target.merged_path)
    assert {"--host", "--port", "--additional-config"} <= set(command)
    assert "--runner" in command and command[command.index("--runner") + 1] == "pooling"


@pytest.mark.parametrize("key", sorted(TARGETS))
def test_release_hashes_cover_weights_and_config(key: str) -> None:
    expected = fetch.expected_files(TARGETS[key].merged_artifact)
    assert "config.json" in expected
    assert any(name.endswith(".safetensors") for name in expected)
    assert all(len(digest) == 64 for digest in expected.values())


def test_verified_requires_every_file_to_match(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("{}")
    good = {"a.json": fetch.file_hash(tmp_path / "a.json")}
    assert fetch.verified(tmp_path, good)
    assert not fetch.verified(tmp_path, {**good, "b.json": "0" * 64})
    assert not fetch.verified(tmp_path, {"a.json": "0" * 64})
