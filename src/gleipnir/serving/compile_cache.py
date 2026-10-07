"""Use vLLM's hash protocol while preserving full serving provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Iterator, MutableMapping
from pathlib import Path

NON_COMPUTE_FIELDS = {
    "name",
    "port",
    "baseline",
    "high_reference",
    "high_concurrency",
    "log_directory",
    "profiler_config",
    "merged_model",
    "config_sha256",
    "startup_audit",
    "allow_finite_parity_diagnostic",
}


def compile_identity(data: dict, root: Path, versions: dict) -> dict:
    """Invalidate on arithmetic, source, receipt contents and runtime changes."""
    condition = {}
    for key, value in data["serving_condition"].items():
        if key in NON_COMPUTE_FIELDS:
            continue
        if key.endswith("_validation") or key.endswith("_reference"):

            def bind(item):
                if isinstance(item, dict):
                    return {k: bind(v) for k, v in item.items()}
                path = (root / item).resolve()
                if not path.is_relative_to(root.resolve() / "results"):
                    raise ValueError("native receipt must be under results")
                return hashlib.sha256(path.read_bytes()).hexdigest()

            value = bind(value)
        condition[key] = value
    sources = {
        path: digest
        for path, digest in data["gleipnir_frost_fp4"].items()
        if not Path(path).name.endswith(("_canary.py", "_compare.py"))
        and Path(path).name not in {"run.py", "serving_runtime.py"}
        and path != "src/gleipnir/serving/runtime.py"
        and not Path(path).name.startswith("prefill_graph_canary")
    }
    return {
        "schema": 1,
        "condition": condition,
        "kernel_sources": sources,
        "versions": versions,
        "other_additional_config": {
            k: v
            for k, v in data.items()
            if k not in {"serving_condition", "gleipnir_frost_fp4"}
        },
    }


class ServingCompileConfig(MutableMapping):
    """Picklable hash-protocol object with unchanged dictionary access."""

    def __init__(self, data: dict, root: Path, versions: dict) -> None:
        self._data = data
        self._root, self._versions = root, versions
        self._serialized = json.dumps(data, sort_keys=True)
        self.identity = compile_identity(data, root, versions)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value) -> None:
        self._data[key] = value

    def __delitem__(self, key) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(self._data)

    def compute_hash(self) -> str:
        serialized = json.dumps(self._data, sort_keys=True)
        if serialized != self._serialized:
            self.identity = compile_identity(self._data, self._root, self._versions)
            self._serialized = serialized
        return hashlib.sha256(
            json.dumps(self.identity, sort_keys=True).encode()
        ).hexdigest()


def install_compile_identity(root: Path) -> None:
    """Wrap parsed engine args, using the upstream SupportsHash protocol."""
    from vllm.engine.arg_utils import AsyncEngineArgs

    if importlib.metadata.version("vllm") != "0.24.0":
        raise ValueError("serving compile identity requires the validated vLLM version")
    original = AsyncEngineArgs.from_cli_args.__func__
    original_create = AsyncEngineArgs.create_engine_config
    versions = {}
    for name in [
        "torch",
        "vllm",
        "transformers",
        "triton",
        "flashinfer-python",
        "nvidia-cutlass-dsl",
        "nvidia-cudnn-cu13",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None

    def parsed(cls, args):
        result = original(cls, args)
        data = result.additional_config
        if (
            isinstance(data, dict)
            and "gleipnir_frost_fp4" in data
            and "serving_condition" in data
        ):
            wrapped = ServingCompileConfig(data, root, versions)
            result.additional_config = wrapped
        return result

    def created(self, *args, **kwargs):
        result = original_create(self, *args, **kwargs)
        wrapped = result.additional_config
        if isinstance(wrapped, ServingCompileConfig):
            digest = wrapped.compute_hash()
            out = root / "results/b200_attention_gdn_serving/compile_identity.json"
            out.write_text(
                json.dumps(
                    {
                        "hash": digest,
                        "identity": wrapped.identity,
                        "full_metadata_preserved": True,
                    },
                    indent=2,
                )
                + "\n"
            )
        return result

    AsyncEngineArgs.from_cli_args = classmethod(parsed)
    AsyncEngineArgs.create_engine_config = created
