"""Dependency relocation preserves caches and compilation follows computation."""

import copy
import json
import pickle

import pytest

from gleipnir.serving_compile_cache import ServingCompileConfig
from gleipnir.serving_runtime import (
    copy_dependency_tree,
    local_serving_runtime,
    runtime_binding,
    sha,
)


def config(root):
    receipt = root / "results/native.json"
    receipt.parent.mkdir()
    receipt.write_text('{"tile":"selected"}')
    return {
        "gleipnir_frost_fp4": {
            "src/gleipnir/kernel.py": "math-source",
            "experiments/screen/kernel_canary.py": "test-source",
        },
        "serving_condition": {
            "name": "first",
            "baseline": "old",
            "high_reference": "old",
            "port": 8010,
            "profiler_config": {"directory": "first"},
            "attention_precision": "mxfp8",
            "worker_cls": "selected",
            "native_validation": "results/native.json",
        },
    }


def test_metadata_and_receipt_location_do_not_recompile_but_provenance_survives(
    tmp_path,
):
    a = config(tmp_path)
    b = copy.deepcopy(a)
    b["serving_condition"].update(
        name="second",
        port=8020,
        baseline="new",
        high_reference="new",
        profiler_config={"directory": "second"},
    )
    other = tmp_path / "results/other.json"
    other.write_bytes((tmp_path / "results/native.json").read_bytes())
    b["serving_condition"]["native_validation"] = "results/other.json"
    b["gleipnir_frost_fp4"]["experiments/screen/kernel_canary.py"] = "new-test"
    x, y = (ServingCompileConfig(v, tmp_path, {"torch": "pinned"}) for v in [a, b])
    assert x.compute_hash() == y.compute_hash()
    assert y["serving_condition"]["name"] == "second"
    assert pickle.loads(pickle.dumps(y)).compute_hash() == y.compute_hash()


@pytest.mark.parametrize(
    "change", ["precision", "source", "receipt", "version", "unknown"]
)
def test_arithmetic_source_receipt_runtime_and_unknown_fields_invalidate(
    tmp_path, change
):
    a = config(tmp_path)
    original = ServingCompileConfig(copy.deepcopy(a), tmp_path, {"torch": "first"})
    before = original.compute_hash()
    versions = {"torch": "first"}
    if change == "precision":
        a["serving_condition"]["attention_precision"] = "bf16"
    elif change == "source":
        a["gleipnir_frost_fp4"]["src/gleipnir/kernel.py"] = "new-math"
    elif change == "receipt":
        (tmp_path / "results/native.json").write_text('{"tile":"changed"}')
    elif change == "version":
        versions["torch"] = "second"
    else:
        a["future_arithmetic_setting"] = 2
    assert ServingCompileConfig(a, tmp_path, versions).compute_hash() != before


def test_mutating_nested_arithmetic_cannot_reuse_previous_hash(tmp_path):
    data = config(tmp_path)
    value = ServingCompileConfig(data, tmp_path, {})
    before = value.compute_hash()
    data["serving_condition"]["attention_precision"] = "bf16"
    assert value.compute_hash() != before


def test_receipt_paths_cannot_read_outside_artifact_tree(tmp_path):
    data = config(tmp_path)
    data["serving_condition"]["native_validation"] = "../private.json"
    with pytest.raises(ValueError, match="under results"):
        ServingCompileConfig(data, tmp_path, {})


def fixture_runtime(root):
    for name in [
        "uv.lock",
        "pyproject.toml",
        ".venv/pyvenv.cfg",
        ".venv/lib/python3.12/site-packages/example.dist-info/METADATA",
    ]:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name)
    return runtime_binding(root)


def test_local_runtime_rewrites_libraries_but_preserves_shared_caches(tmp_path):
    root, target = tmp_path / "source", tmp_path / "local"
    binding = fixture_runtime(root)
    python = target / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("stub")
    manifest = target / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "binding": binding,
                "python": str(python),
                "path_mappings": {str(root / ".venv"): str(target / "venv")},
            }
        )
    )
    env = {
        "GLEIPNIR_SERVING_RUNTIME": str(target),
        "PATH": f"{root}/.venv/bin:/usr/bin",
        "PYTHONPATH": f"{root}/src",
        "TRITON_CACHE_DIR": f"{root}/.cache/shared",
    }
    receipt = local_serving_runtime(root, env)
    assert env["PATH"] == f"{target}/venv/bin:/usr/bin"
    assert env["PYTHONPATH"] == f"{root}/src"
    assert env["TRITON_CACHE_DIR"] == f"{root}/.cache/shared"
    assert receipt["manifest_sha256"] == sha(manifest)
    (root / "uv.lock").write_text("new dependencies")
    with pytest.raises(ValueError, match="stale"):
        local_serving_runtime(root, env)


def test_absent_runtime_falls_back_without_mutating_environment(tmp_path):
    env = {"GLEIPNIR_SERVING_RUNTIME": str(tmp_path), "PATH": "existing"}
    assert local_serving_runtime(tmp_path, env) is None
    assert env["PATH"] == "existing"


def test_parallel_staging_dereferences_links_and_removes_obsolete_packages(tmp_path):
    import shutil

    if shutil.which("rsync") is None:
        pytest.skip("rsync unavailable")
    source, target = tmp_path / "source", tmp_path / "target"
    lib = "lib/python3.12/site-packages"
    old = target / lib / "obsolete"
    old.mkdir(parents=True)
    (old / "unwanted.py").write_text("old dependency")
    native = source / lib / "torch/lib/native.so"
    native.parent.mkdir(parents=True)
    native.write_bytes(b"native library")
    alias = native.parent / "alias.so"
    alias.symlink_to(native)
    (source / "lib64").symlink_to(source / "lib", target_is_directory=True)
    copy_dependency_tree(source, target, workers=2)
    copied = target / lib / "torch/lib/alias.so"
    assert copied.read_bytes() == b"native library" and not copied.is_symlink()
    assert not old.exists() and not (target / "lib64").exists()
