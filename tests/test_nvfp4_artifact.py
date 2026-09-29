"""Native artifacts cannot silently change provenance, weights, or layouts."""

import hashlib
import json

import pytest
import torch

from gleipnir.nvfp4_artifact import NvFp4Artifact, tensor_sha256


def fixture_artifact(root):
    weight = torch.ones(32, 64, dtype=torch.bfloat16)
    path = root / "weight.pt"
    torch.save(
        {
            "packed": torch.zeros(32, 32, dtype=torch.uint8),
            "blocks": torch.ones(32, 4).to(torch.float8_e4m3fn),
            "global_scale": torch.ones(1),
        },
        path,
    )
    entry = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "shape_nk": list(weight.shape),
        "original_weight_sha256": tensor_sha256(weight),
    }
    manifest = {
        "state": "complete",
        "merge_manifest_sha256": "merge",
        "layers": {
            f"{i}_{name}": entry for i in range(32) for name in ("gate_up", "down")
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return NvFp4Artifact(root, digest, "merge"), weight, digest


def test_artifact_validates_original_precision_and_native_layout(tmp_path):
    artifact, weight, _ = fixture_artifact(tmp_path)
    packed, blocks, scale = artifact.load("model.layers.0.mlp.down_proj", weight)
    assert packed.dtype == torch.uint8
    assert blocks.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    with pytest.raises(ValueError, match="source weight"):
        artifact.load("model.layers.0.mlp.down_proj", weight + 1)


def test_artifact_rejects_changed_manifest_or_source_provenance(tmp_path):
    _, _, digest = fixture_artifact(tmp_path)
    with pytest.raises(ValueError, match="manifest"):
        NvFp4Artifact(tmp_path, "wrong", "merge")
    with pytest.raises(ValueError, match="provenance"):
        NvFp4Artifact(tmp_path, digest, "different merge")


def test_artifact_rejects_changed_packed_file(tmp_path):
    artifact, weight, _ = fixture_artifact(tmp_path)
    (tmp_path / "weight.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="packed tensor"):
        artifact.load("model.layers.0.mlp.down_proj", weight)
