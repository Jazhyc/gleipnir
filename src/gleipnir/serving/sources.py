"""Locate live implementations named by source-bound serving receipts."""

from pathlib import Path

from gleipnir._compat import canonical_source_reference


def recorded_source_path(root: Path, reference: str) -> Path:
    """Resolve a moved project source while preserving external/archive paths.

    Callers must still compare its bytes with the receipt's original checksum.
    This changes neither the receipt nor the location of archived evidence.
    """
    root = root.resolve()
    path = Path(reference)
    if path.is_absolute():
        if not path.is_relative_to(root):
            return path
        reference = path.relative_to(root).as_posix()
    return root / canonical_source_reference(reference)
