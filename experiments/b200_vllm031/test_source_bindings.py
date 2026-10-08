"""Verify source restoration cannot waive an executable or archived checksum."""

import hashlib
import io
import json
import tarfile

import pytest

from experiments.b200_vllm031.source_bindings import restore_diagnostic_sources


def fixture(tmp_path, relative, archived=b"validated", expected=None):
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"new repository copy")
    receipt = tmp_path / "results/receipt.json"
    receipt.parent.mkdir()
    receipt.write_text(
        json.dumps(
            {
                "sources": {
                    relative: expected or hashlib.sha256(archived).hexdigest(),
                }
            }
        )
    )
    archive = tmp_path / "original.tar.gz"
    with tarfile.open(archive, "w:gz") as saved:
        info = tarfile.TarInfo(relative)
        info.size = len(archived)
        saved.addfile(info, io.BytesIO(archived))
    return path, archive


def test_restore_exact_diagnostic_source(tmp_path):
    path, archive = fixture(tmp_path, "experiments/example/check_canary.py")
    restored = restore_diagnostic_sources(
        tmp_path, {"native_validation": "results/receipt.json"}, archive
    )
    assert path.read_bytes() == b"validated" and len(restored) == 1


def test_runtime_source_mismatch_requires_new_validation(tmp_path):
    path, archive = fixture(tmp_path, "experiments/example/worker.py")
    with pytest.raises(ValueError, match="runtime source drift"):
        restore_diagnostic_sources(
            tmp_path, {"native_validation": "results/receipt.json"}, archive
        )
    assert path.read_bytes() == b"new repository copy"


def test_archive_must_match_frozen_checksum(tmp_path):
    path, archive = fixture(
        tmp_path, "experiments/example/check_canary.py", expected="0" * 64
    )
    with pytest.raises(ValueError, match="archived diagnostic checksum"):
        restore_diagnostic_sources(
            tmp_path, {"native_validation": "results/receipt.json"}, archive
        )
    assert path.read_bytes() == b"new repository copy"
