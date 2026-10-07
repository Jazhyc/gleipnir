"""Serving source relocation preserves frozen evidence and lazy dependencies."""

import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.b200_attention_gdn_serving.cache_policy_compare import _live_sources
from experiments.b200_inference_benchmark.run import compatible_server_command
from experiments.fp4_inference.run import resolve_runtime_config
from gleipnir.serving.compile_cache import ServingCompileConfig
from gleipnir.serving.sources import recorded_source_path


@pytest.mark.parametrize("absolute", [False, True])
def test_recorded_live_sources_resolve_old_and_current_paths(tmp_path, absolute):
    old = "src/gleipnir/serving_runtime.py"
    current = "src/gleipnir/serving/runtime.py"
    references = [str(tmp_path / p) if absolute else p for p in [old, current]]
    for reference in references:
        assert recorded_source_path(tmp_path, reference) == tmp_path / current


def test_archived_and_external_source_paths_are_not_translated(tmp_path):
    archived = "results/old/executed_sources/src/gleipnir/serving_runtime.py"
    external = tmp_path.parent / "vendor/src/gleipnir/serving_runtime.py"
    assert recorded_source_path(tmp_path, archived) == tmp_path / archived
    assert recorded_source_path(tmp_path, str(external)) == external


def test_frozen_config_is_unchanged_and_new_identity_binds_live_sources():
    config = {
        "engine": {"quantization": "gleipnir_nvfp4"},
        "additional_code_files": ["src/gleipnir/nvfp4_reference.py"],
    }
    resolved = resolve_runtime_config(config)
    assert config == {
        "engine": {"quantization": "gleipnir_nvfp4"},
        "additional_code_files": ["src/gleipnir/nvfp4_reference.py"],
    }
    paths = resolved["additional_code_files"]
    assert paths == [
        "src/gleipnir/kernels/fp4/nvfp4_reference.py",
        "src/gleipnir/__init__.py",
        "src/gleipnir/_compat.py",
    ]
    identity = resolved["engine"]["additional_config"]["gleipnir_nvfp4"]
    assert set(identity["additional_source_sha256"]) == set(paths)
    assert all(Path(p).is_file() for p in paths)


def test_serving_and_kernel_packages_do_not_load_gpu_dependencies():
    code = """
import sys
import gleipnir.serving
import gleipnir.serving.fp4
import gleipnir.serving.gdn
import gleipnir.serving.vllm
import gleipnir.kernels
import gleipnir.kernels.fp4
import gleipnir.kernels.mxfp8
import gleipnir.training.fp4
assert not {'torch', 'triton', 'vllm', 'cutlass', 'cudnn'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_relocated_launcher_is_still_excluded_from_compute_identity(tmp_path):
    import json

    def config(path, digest):
        return {
            "gleipnir_frost_fp4": {
                path: digest,
                "src/gleipnir/kernels/fp4/cudnn_fp4_gemm.py": "same-arithmetic",
            },
            "serving_condition": {"attention_precision": "bf16"},
        }

    old = config("src/gleipnir/serving_runtime.py", "old-launcher")
    new = config("src/gleipnir/serving/runtime.py", "new-launcher")
    assert ServingCompileConfig(old, tmp_path, {}).compute_hash() == (
        ServingCompileConfig(new, tmp_path, {}).compute_hash()
    )
    commands = [
        ["python", "-m", "server", "--additional-config", json.dumps(value)]
        for value in (old, new)
    ]
    assert compatible_server_command(*commands)


def test_historical_compile_config_pickle_resolves_canonical_class():
    assert (
        pickle.loads(b"cgleipnir.serving_compile_cache\nServingCompileConfig\n.")
        is ServingCompileConfig
    )


def test_deployed_snapshot_paths_are_preserved_and_current_paths_are_resolved(tmp_path):
    original = "src/gleipnir/serving_runtime.py"
    source = tmp_path / original
    source.parent.mkdir(parents=True)
    source.write_text("original deployed bytes")
    assert _live_sources(tmp_path, {original}) == {original}
    source.unlink()
    assert _live_sources(tmp_path, {original}) == {
        "src/gleipnir/serving/runtime.py",
        "src/gleipnir/__init__.py",
        "src/gleipnir/_compat.py",
    }
