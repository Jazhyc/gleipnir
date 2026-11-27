"""Locate live implementations named by source-bound serving receipts.

Receipts and accepted recipes record sources as ``path: sha256``. Sources moved
since then resolve through ``_compat`` aliases. Runtime files promoted into
``gleipnir.serving.stack``, and a few sources rewritten in place, also changed
bytes; the promotion manifest (``stack/promotion.json``) records each one's
recorded path and hash before the change and its path and hash after it, so a
recorded ``(path, hash)`` is satisfied exactly when the hash is the one the
manifest saw before the change and the live file still has its manifest hash.
"""

from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path

from gleipnir._compat import MODULE_ALIASES, canonical_source_reference

MANIFEST = Path(__file__).with_name("stack") / "promotion.json"


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


def _relative(root: Path, reference: str) -> str:
    path = Path(reference)
    if path.is_absolute():
        root = root.resolve()
        if path.is_relative_to(root):
            return path.relative_to(root).as_posix()
        for prefix in ("/workspace/gleipnir/", "/home/ubuntu/gleipnir/"):
            if reference.startswith(prefix):
                return reference.removeprefix(prefix)
    return reference


@cache
def promotion() -> dict:
    """The promotion manifest: ``{"status", "sources": {recorded path: entry}}``."""
    return json.loads(MANIFEST.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_source_matches(root: Path, reference: str, digest: str) -> bool:
    """Whether a recorded ``(reference, digest)`` still describes live code."""
    path = recorded_source_path(root, reference)
    if path.is_file() and _sha(path) == digest:
        return True
    moved = promotion()["sources"].get(_relative(root, reference))
    return (
        moved is not None
        and digest == moved["sha256_before"]
        and path.is_file()
        and _sha(path) == moved["sha256"]
    )


def require_recorded_source(root: Path, reference: str, digest: str, what: str) -> None:
    if not recorded_source_matches(root, reference, digest):
        raise ValueError(f"{what} source drift: {reference}")


def is_diagnostic(reference: str) -> bool:
    """Sources the serving gate does not bind: tests, canaries and runners."""
    name = Path(reference).name
    return (
        name.startswith(("test_", "prefill_graph_canary"))
        or name.endswith(("_canary.py", "_compare.py"))
        or name in {"run.py", "fp4_gemm_tune.py", "serving_runtime.py"}
        or reference == "src/gleipnir/serving/reference.py"
    )


def promoted_sources(root: Path, sources: dict[str, str]) -> dict[str, str]:
    """Translate a recorded source map to live paths and hashes.

    Entries that still describe live code move to their live path and hash.
    Others (diagnostics, or sources a launch gate does not bind) keep their
    recorded entry unchanged; verifying them is the gates' job, not this one's.
    """
    live = {}
    for reference, digest in sources.items():
        if recorded_source_matches(root, reference, digest):
            path = recorded_source_path(root, reference)
            live[path.relative_to(root.resolve()).as_posix()] = _sha(path)
        else:
            live[reference] = digest
    return live


def promoted_module(name: str) -> str:
    """The live module (or ``module.Class``) for a recorded experiment name."""
    module, _, attribute = name.rpartition(".")
    if name in MODULE_ALIASES:
        return MODULE_ALIASES[name]
    if module in MODULE_ALIASES:
        return f"{MODULE_ALIASES[module]}.{attribute}"
    return name


def _promote_strings(value):
    if isinstance(value, dict):
        return {k: _promote_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_promote_strings(v) for v in value]
    if isinstance(value, str) and value.startswith("experiments."):
        return promoted_module(value)
    return value


def promote_command(root: Path, command: list[str]) -> list[str]:
    """Rewrite a verified recorded server command onto the promoted stack.

    Module names (``-m``, ``--worker-cls``, ``--scheduler-cls`` and names inside
    ``--additional-config``) move to their live modules; the
    ``gleipnir_frost_fp4`` source map moves to live paths and hashes after each
    recorded entry is verified. Everything else is unchanged.
    """
    promoted = [_promote_strings(a) if not a.startswith("{") else a for a in command]
    if "--additional-config" in promoted:
        index = promoted.index("--additional-config") + 1
        additional = _promote_strings(json.loads(promoted[index]))
        if "gleipnir_frost_fp4" in additional:
            additional["gleipnir_frost_fp4"] = promoted_sources(
                root, additional["gleipnir_frost_fp4"]
            )
        promoted[index] = json.dumps(additional, sort_keys=True)
    return promoted


SCHEDULER = "experiments/b200_vllm031/scheduler.py"


def scheduler_binding(root: Path, recorded: dict) -> dict:
    """Verify a recorded pooling-scheduler binding and return the live one."""
    import inspect

    from vllm.v1.core.sched import scheduler

    upstream = _sha(Path(inspect.getfile(scheduler)))
    if upstream != recorded["upstream_sha256"]:
        raise ValueError("selected pooling scheduler changed: upstream")
    require_recorded_source(
        root, SCHEDULER, recorded["integration_sha256"], "selected pooling scheduler"
    )
    return {
        "upstream_sha256": upstream,
        "integration_sha256": _sha(recorded_source_path(root, SCHEDULER)),
    }


def build_manifest(root: Path, moves: list[tuple[str, str, str]], status: str) -> dict:
    """Manifest entries from ``(old path, new path, sha256 before)`` triples."""
    return {
        "status": status,
        "sources": {
            old: {"path": new, "sha256_before": before, "sha256": _sha(root / new)}
            for old, new, before in sorted(moves)
        },
    }
