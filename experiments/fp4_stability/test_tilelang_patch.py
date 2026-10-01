"""Verify fail-closed source identity and idempotence of the compiler patch."""

import hashlib
import json

import pytest

from experiments.fp4_stability import patch_tilelang_fp16 as patch


def test_patch_rejects_unrecognized_header():
    with pytest.raises(ValueError, match="unexpected original"):
        patch.patched_header(b"unrecognized vendor source")


def test_recorded_patch_is_idempotent_and_rejects_later_drift(tmp_path, monkeypatch):
    source = (
        "__device__ __forceinline__ unsigned long long\n"
        + patch.SIGNATURE
        + "\n  return 0;\n}\n"
    ).encode()
    monkeypatch.setattr(patch, "ORIGINAL_SHA256", hashlib.sha256(source).hexdigest())
    monkeypatch.setattr(patch, "TARGET", tmp_path)
    header = tmp_path / patch.HEADER
    header.parent.mkdir(parents=True)
    header.write_bytes(source)
    manifest = tmp_path / "install_manifest.json"
    manifest.write_text("{}")
    patch.main()
    recorded = json.loads(manifest.read_text())["compiler_patches"][0]
    assert recorded["before_sha256"] == hashlib.sha256(source).hexdigest()
    assert recorded["after_sha256"] == hashlib.sha256(header.read_bytes()).hexdigest()
    assert patch.SIGNATURE.encode() in header.read_bytes()
    assert b"const cutlass::half_t" in header.read_bytes()
    unchanged = header.read_bytes(), manifest.read_bytes()
    patch.main()
    assert unchanged == (header.read_bytes(), manifest.read_bytes())
    header.write_bytes(header.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="header drift"):
        patch.main()
