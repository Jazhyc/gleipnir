"""Selected CPU serving components, separate from GPU compilation identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def selected_host_components(
    root: Path, output: Path, *, resident: dict | None = None
) -> tuple[dict | None, dict | None]:
    """Bind selected receipts and preserve the actual resident frontend path."""
    selection = json.loads(
        (root / "experiments/b200_inference_benchmark/baseline.json").read_text()
    )
    frontend = selection.get("frontend")
    host = selection.get("host_wrapper")
    for component in (frontend, host):
        if component is not None:
            path = root / component["validation"]
            if not path.resolve().is_relative_to(root.resolve() / "results"):
                raise ValueError("selected host validation must be an artifact")
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != component["validation_sha256"]
            ):
                raise ValueError("selected host validation checksum drift")
    if frontend is not None:
        frontend = {k: v for k, v in frontend.items() if k != "validation_sha256"}
        frontend["receipt_path"] = (
            resident["frontend"]["receipt_path"]
            if resident and resident.get("frontend")
            else str(output / "frontend.json")
        )
    if host is not None:
        host = {"validation": host["validation"]}
    return frontend, host
