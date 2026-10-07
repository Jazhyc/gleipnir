"""Bundle restoration preserves dependency identity and existing workers."""

import io
import json
import tarfile

import pytest

from gleipnir.serving_bundle import (
    digest,
    extract_archive,
    restore_bundle,
    verify_bundle,
)
from gleipnir.serving_runtime import local_serving_runtime


def archive(path, files):
    with tarfile.open(path, "w:gz") as target:
        for name, value in files.items():
            data = value.encode()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            target.addfile(member, io.BytesIO(data))


def fixture_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source = {"uv.lock": "locked", "pyproject.toml": "project"}
    bound = {
        **source,
        ".venv/pyvenv.cfg": "python",
        ".cache/kernels/fa4/.git/HEAD": "commit",
    }
    runtime = {
        "gleipnir-serving-runtime/venv/pyvenv.cfg": "python",
        "gleipnir-serving-runtime/venv/bin/python": "interpreter",
        "gleipnir-gigatoken-0.10.0/gigatoken/__init__.py": "native",
    }
    archive(bundle / "runtime.tar.gz", runtime)
    archive(bundle / "source.tar.gz", source)
    archive(bundle / "context.tar.gz", {"results/parity.json": "passed"})
    archive(bundle / "caches.tar.gz", {".cache/training/shared/key": "compiled"})
    (bundle / "binding.json").write_text(json.dumps(bound))
    import hashlib

    receipt = {
        "schema": 1,
        "binding": {
            "source_root": "/originalproject",
            "files": {
                name: hashlib.sha256(value.encode()).hexdigest()
                for name, value in bound.items()
            },
        },
        "python": "/tmp/gleipnir-serving-runtime/venv/bin/python",
        "path_mappings": {
            "/originalproject/.venv": "/tmp/gleipnir-serving-runtime/venv",
            "/originalproject/.cache/kernels/fa4": "/tmp/gleipnir-serving-runtime/fa4",
        },
    }
    (bundle / "runtime.json").write_text(json.dumps(receipt))
    manifest = {
        "schema": 1,
        "archives": {
            name: name + ".tar.gz"
            for name in ["runtime", "source", "context", "caches"]
        },
        "metadata": {
            "runtime_manifest": "runtime.json",
            "binding_source_files": "binding.json",
        },
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in bundle.iterdir()
        },
    }
    (bundle / "bundle.json").write_text(json.dumps(manifest))
    return bundle


def test_restore_is_admitted_by_ordinary_launcher_and_preserves_shared_cache(tmp_path):
    bundle = fixture_bundle(tmp_path)
    root, runtime, tokenizer = (
        tmp_path / n for n in ["project", "runtime", "tokenizer"]
    )
    restore_bundle(bundle, root, runtime, tokenizer)
    env = {
        "GLEIPNIR_SERVING_RUNTIME": str(runtime),
        "PATH": f"{root}/.venv/bin:/usr/bin",
    }
    receipt = local_serving_runtime(root, env)
    assert receipt["python"] == str(runtime / "venv/bin/python")
    assert env["PATH"] == f"{runtime}/venv/bin:/usr/bin"
    assert (root / ".cache/training/shared/key").read_text() == "compiled"
    assert (runtime / "fa4/.git/HEAD").read_text() == "commit"
    assert (root / "results/parity.json").read_text() == "passed"


def test_checksum_failure_precedes_all_destination_writes(tmp_path):
    bundle = fixture_bundle(tmp_path)
    (bundle / "source.tar.gz").write_bytes(b"changed")
    targets = [tmp_path / n for n in ["project", "runtime", "tokenizer"]]
    with pytest.raises(ValueError, match="checksum"):
        restore_bundle(bundle, *targets)
    assert not any(p.exists() for p in targets)


def test_resident_runtime_cannot_be_overwritten(tmp_path):
    bundle = fixture_bundle(tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "resident").write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        restore_bundle(bundle, tmp_path / "project", runtime, tmp_path / "tokenizer")
    assert (runtime / "resident").read_text() == "keep"
    assert not (tmp_path / "project").exists()


def test_valid_archive_checksum_cannot_hide_dependency_drift(tmp_path):
    bundle = fixture_bundle(tmp_path)
    path = bundle / "source.tar.gz"
    archive(path, {"uv.lock": "different lock", "pyproject.toml": "project"})
    manifest_path = bundle / "bundle.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][path.name] = {
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }
    manifest_path.write_text(json.dumps(manifest))
    targets = [tmp_path / n for n in ["project", "runtime", "tokenizer"]]
    with pytest.raises(ValueError, match="dependency mismatch"):
        restore_bundle(bundle, *targets)
    assert not any(p.exists() for p in targets)


def test_archive_cannot_escape_destination(tmp_path):
    path = tmp_path / "bad.tar.gz"
    archive(path, {"../outside": "escape"})
    with pytest.raises(tarfile.OutsideDestinationError):
        extract_archive(path, tmp_path / "target")
    assert not (tmp_path / "outside").exists()


def test_bundle_reference_must_be_checksum_bound(tmp_path):
    bundle = fixture_bundle(tmp_path)
    path = bundle / "bundle.json"
    manifest = json.loads(path.read_text())
    manifest["archives"]["source"] = "unverified.tar.gz"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unbound"):
        verify_bundle(bundle)
