"""Native tokenizer restoration verifies archived bytes before registration."""

import hashlib
import tarfile
from pathlib import Path

import pytest

from gleipnir.serving.tokenizer_setup import prepare_native_tokenizer


def payload(tmp_path: Path):
    source = tmp_path / "source/tokenizer/gigatoken"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("VERSION = '0.10.0'\n")
    archive = tmp_path / "tokenizer.tar.gz"
    with tarfile.open(archive, "w:gz") as contents:
        contents.add(source.parent, arcname="tokenizer")
    expected = {
        "gigatoken/__init__.py": hashlib.sha256(
            (source / "__init__.py").read_bytes()
        ).hexdigest()
    }
    site = tmp_path / "site-packages"
    site.mkdir()
    return tmp_path / "runtime/tokenizer", archive, [site], expected


def test_fresh_restore_and_repeated_registration_are_idempotent(tmp_path):
    args = payload(tmp_path)
    first = prepare_native_tokenizer(*args)
    second = prepare_native_tokenizer(*args)
    assert first["restored"] and not second["restored"]
    assert Path(first["registrations"][0]).read_text() == str(args[0]) + "\n"


def test_bad_archive_bytes_leave_no_installed_package(tmp_path):
    package, archive, sites, expected = payload(tmp_path)
    expected["gigatoken/__init__.py"] = "wrong"
    with pytest.raises(ValueError, match="package drift"):
        prepare_native_tokenizer(package, archive, sites, expected)
    assert not package.exists()
    assert not (sites[0] / "gleipnir_native_gigatoken.pth").exists()


def test_existing_drift_is_preserved_for_diagnosis(tmp_path):
    package, archive, sites, expected = payload(tmp_path)
    prepare_native_tokenizer(package, archive, sites, expected)
    changed = package / "gigatoken/__init__.py"
    changed.write_text("different\n")
    with pytest.raises(ValueError, match="package drift"):
        prepare_native_tokenizer(package, archive, sites, expected)
    assert changed.read_text() == "different\n"


def test_unsafe_receipt_paths_are_rejected_before_restoration(tmp_path):
    package, archive, sites, _ = payload(tmp_path)
    with pytest.raises(ValueError, match="unsafe"):
        prepare_native_tokenizer(package, archive, sites, {"../escape": "wrong"})
    assert not package.exists()
