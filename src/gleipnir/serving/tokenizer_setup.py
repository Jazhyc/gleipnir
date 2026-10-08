"""Restore and register the checksum-bound native tokenizer for fresh runtimes."""

from __future__ import annotations

import hashlib
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path


def prepare_native_tokenizer(
    package: Path,
    archive: Path,
    sites: Sequence[Path],
    expected_files: Mapping[str, str],
) -> dict:
    """Restore missing files atomically and persist an idempotent path registration."""
    for name in expected_files:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe native tokenizer file path")

    def verify(root: Path) -> None:
        for name, expected in expected_files.items():
            path = root / name
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise ValueError(f"native tokenizer package drift: {name}")

    restored = False
    if not package.exists():
        if not archive.is_file():
            raise ValueError(f"restore the native tokenizer archive first: {archive}")
        package.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=package.parent) as temporary:
            staging = Path(temporary)
            with tarfile.open(archive) as contents:
                contents.extractall(staging, filter="data")
            extracted = staging / package.name
            verify(extracted)
            extracted.rename(package)
        restored = True
    verify(package)
    registrations = []
    for site in sites:
        if not site.is_dir():
            raise ValueError(f"candidate site-packages missing: {site}")
        path = site / "gleipnir_native_gigatoken.pth"
        value = str(package.resolve()) + "\n"
        if path.exists() and path.read_text() != value:
            raise ValueError(f"preserve existing native tokenizer registration: {path}")
        path.write_text(value)
        registrations.append(str(path))
    return {
        "package": str(package),
        "archive": str(archive),
        "restored": restored,
        "verified_files": len(expected_files),
        "registrations": registrations,
    }
