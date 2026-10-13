"""Only merged shards disappear; reconstruction sources and receipts survive."""

import json
import struct

import pytest

from gleipnir.data.monitoring import file_hash
from scripts import prune_reconstructable_merges as cleanup


def fixture_merge(root, dtype="F32"):
    adapter = root / "results/source_adapter"
    adapter.mkdir(parents=True)
    h = json.dumps({"tensor": {"dtype": dtype}}).encode()
    (adapter / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(h)) + h)
    (adapter / "adapter_config.json").write_text("{}")
    base = root / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/revision"
    base.mkdir(parents=True)
    (base / "config.json").write_text("{}")
    checkpoint = root / cleanup.ROOTS[0] / "merged"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"merged")
    manifest = {
        "adapter": str(adapter),
        "adapter_sha256": file_hash(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": file_hash(adapter / "adapter_config.json"),
        "model": "Qwen/Qwen3.5-4B",
        "revision": "revision",
        "source_files_sha256": {"config.json": file_hash(base / "config.json")},
        "files_sha256": {
            "model.safetensors": file_hash(checkpoint / "model.safetensors")
        },
    }
    (checkpoint / "merge_manifest.json").write_text(json.dumps(manifest))
    return adapter, base, checkpoint


def test_delete_preserves_sources_and_metadata(monkeypatch, tmp_path):
    adapter, base, checkpoint = fixture_merge(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["cleanup", "--execute"])
    cleanup.main()
    assert not (checkpoint / "model.safetensors").exists()
    assert (checkpoint / "merge_manifest.json").exists()
    assert (checkpoint / "weights_pruned.json").exists()
    assert (adapter / "adapter_model.safetensors").exists()
    assert (base / "config.json").exists()


@pytest.mark.parametrize("failure", ["adapter", "base", "precision"])
def test_bad_reconstruction_source_rejects_plan(failure, tmp_path):
    adapter, base, checkpoint = fixture_merge(
        tmp_path, "BF16" if failure == "precision" else "F32"
    )
    if failure == "adapter":
        (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    if failure == "base":
        (base / "config.json").write_text("changed")
    with pytest.raises(ValueError):
        cleanup.candidates(tmp_path)
    assert (checkpoint / "model.safetensors").exists()
