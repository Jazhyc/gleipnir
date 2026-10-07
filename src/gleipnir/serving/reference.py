"""Selected CPU serving components, separate from GPU compilation identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def selected_score_reference(root: Path) -> dict:
    """Verify the selected score workload, receipts and every saved repeat."""
    selection = json.loads(
        (root / "experiments/b200_inference_benchmark/baseline.json").read_text()
    )
    if selection.get("endpoint") != "/v1/monitor/score":
        raise ValueError("selected reference is not a monitoring score server")
    for path, digest in selection["artifact_bindings"].items():
        artifact = root / path
        if not artifact.resolve().is_relative_to(root.resolve() / "results") or (
            hashlib.sha256(artifact.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("selected score reference artifact drift")
    directory = root / selection["results"]
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "complete" or not summary["score_audit_passed"]:
        raise ValueError("selected score reference incomplete")
    manifest = root / "data/b200_inference_benchmark/manifest.json"
    if (
        hashlib.sha256(manifest.read_bytes()).hexdigest()
        != selection["manifest_sha256"]
    ):
        raise ValueError("selected score reference workload drift")
    expected = {
        str((directory / f"c{c}_repeat{i}.json").relative_to(root))
        for c, count in ((1, 3), (128, 6))
        for i in range(count)
    }
    if not expected.issubset(selection["artifact_bindings"]):
        raise ValueError("selected score reference repeats unbound")
    return selection


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
