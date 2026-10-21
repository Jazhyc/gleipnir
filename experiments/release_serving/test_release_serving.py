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


def _long_context_case(tmp_path: Path, key: str):
    import dataclasses

    from experiments.release_serving.serve import extend_context

    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    source = tmp_path / "worker.py"
    source.write_text("pass\n")
    target = TARGETS[key]
    spec = dataclasses.replace(
        target.long_context,
        receipt=str(receipt),
        receipt_sha256=fetch.file_hash(receipt),
        sources=(str(source),) if target.long_context.sources else (),
    )
    target = dataclasses.replace(target, long_context=spec)
    command = json.loads((ROOT / TARGETS[key].server_receipt).read_text())["command"]
    return extend_context, target, command


@pytest.mark.parametrize("key", sorted(TARGETS))
def test_recorded_context_replays_the_recorded_command(
    key: str, tmp_path: Path
) -> None:
    extend_context, target, command = _long_context_case(tmp_path, key)
    original, env = list(command), {}
    extend_context(command, env, target, 32768)
    assert command == original and env == {}


@pytest.mark.parametrize("key", sorted(TARGETS))
def test_long_context_changes_only_limit_worker_and_receipt(
    key: str, tmp_path: Path
) -> None:
    extend_context, target, command = _long_context_case(tmp_path, key)
    original, env = list(command), {}
    extend_context(command, env, target, 262144)
    changed = {
        original[i - 1]
        for i, (a, b) in enumerate(zip(original, command, strict=True))
        if a != b
    }
    assert changed <= {"--max-model-len", "--worker-cls", "--additional-config"}
    assert command[command.index("--max-model-len") + 1] == "262144"
    assert command[command.index("--max-num-batched-tokens") + 1] == "32768"
    before = json.loads(original[original.index("--additional-config") + 1])
    after = json.loads(command[command.index("--additional-config") + 1])
    spec = target.long_context
    if spec.receipt_env:
        assert env == {spec.receipt_env: spec.receipt}
        assert command[command.index("--worker-cls") + 1] == spec.worker[1]
        assert set(after["gleipnir_frost_fp4"]) - set(
            before["gleipnir_frost_fp4"]
        ) == set(spec.sources)
    else:
        assert env == {} and "--worker-cls" not in changed
        assert after[spec.additional_key]["validation_sha256"] == spec.receipt_sha256
    assert {k: v for k, v in after.items() if k != spec.additional_key} == {
        **before,
        "gleipnir_frost_fp4": after["gleipnir_frost_fp4"],
    }


def test_long_context_rejects_a_changed_receipt(tmp_path: Path) -> None:
    import dataclasses

    extend_context, target, command = _long_context_case(tmp_path, "4b")
    stale = dataclasses.replace(target.long_context, receipt_sha256="0" * 64)
    with pytest.raises(ValueError, match="receipt changed"):
        extend_context(
            command, {}, dataclasses.replace(target, long_context=stale), 262144
        )
    with pytest.raises(ValueError, match="context limit"):
        extend_context(command, {}, target, 2 * 262144)
