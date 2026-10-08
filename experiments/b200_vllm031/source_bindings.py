"""Restore checksum-bound diagnostic generators without replacing runtime code."""

import hashlib
import json
import tarfile
from pathlib import Path

from gleipnir._compat import canonical_source_reference


def restore_diagnostic_sources(
    root: Path, condition: dict, archive: Path
) -> list[dict]:
    """Require exact archived bytes for diagnostics referenced by frozen receipts."""
    paths = []
    for key, value in condition.items():
        if key.endswith("_validation") or key.endswith("_reference"):
            paths.extend(value.values() if isinstance(value, dict) else [value])
    restored = []
    for reference in paths:
        if not str(reference).startswith("results/"):
            continue
        receipt = json.loads((root / reference).read_text())
        for source, expected in receipt.get("sources", {}).items():
            relative = source.removeprefix("/workspace/gleipnir/")
            path = root / canonical_source_reference(relative)
            if not path.resolve().is_relative_to(root.resolve()):
                continue  # External dependencies retain their independent checks.
            if hashlib.sha256(path.read_bytes()).hexdigest() == expected:
                continue
            if not relative.startswith("experiments/") or not (
                path.name.endswith(("_canary.py", "_compare.py"))
                or path.name == "fp4_gemm_tune.py"
            ):
                raise ValueError(
                    f"runtime source drift requires validation: {relative}"
                )
            with tarfile.open(archive, "r:gz") as saved:
                entry = saved.extractfile(relative)
                if entry is None:
                    raise ValueError(f"archived diagnostic is missing: {relative}")
                data = entry.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"archived diagnostic checksum differs: {relative}")
            previous = hashlib.sha256(path.read_bytes()).hexdigest()
            path.write_bytes(data)
            restored.append(
                {
                    "source": relative,
                    "previous_sha256": previous,
                    "restored_sha256": expected,
                    "receipt": reference,
                }
            )
    return restored
