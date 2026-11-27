"""The promoted serving runtime keeps every accepted recipe's identity."""

import hashlib
import importlib
import json
import subprocess
import sys

from gleipnir._compat import MODULE_ALIASES
from gleipnir.serving.sources import (
    promote_command,
    promoted_module,
    promotion,
    recorded_source_matches,
)
from tests.helpers.paths import ROOT


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_matches_the_live_promoted_files():
    manifest = promotion()
    assert manifest["status"] in {"candidate", "accepted"}
    for recorded, entry in manifest["sources"].items():
        live = ROOT / entry["path"]
        assert _sha(live) == entry["sha256"], f"edit invalidates promotion: {live}"
        legacy = recorded.removesuffix(".py").replace("/", ".")
        if recorded.startswith("experiments/") and recorded != entry["path"]:
            assert MODULE_ALIASES[legacy] == entry["path"].removeprefix(
                "src/"
            ).removesuffix(".py").replace("/", ".")


def test_recorded_pre_move_hash_is_satisfied_only_through_the_manifest():
    recorded, entry = next(
        (k, v) for k, v in promotion()["sources"].items() if k != v["path"]
    )
    assert recorded_source_matches(ROOT, recorded, entry["sha256_before"])
    assert not recorded_source_matches(ROOT, recorded, "0" * 64)


def test_promote_command_renames_modules_and_rebinds_sources():
    recorded, entry = next(
        (k, v)
        for k, v in promotion()["sources"].items()
        if k.endswith("gdn_fast/worker.py")
    )
    command = [
        "python",
        "-m",
        "experiments.b200_attention_precision.server",
        "--worker-cls",
        "experiments.b200_gdn_fast.worker.FastGdnWorker",
        "--additional-config",
        json.dumps(
            {
                "gleipnir_frost_fp4": {recorded: entry["sha256_before"], "x.py": "1"},
                "serving_condition": {
                    "server_module": "experiments.b200_attention_gdn_serving.server"
                },
            }
        ),
        "--hf-overrides",
        '{"keep": "experiments.not_a_module"}',
    ]
    promoted = promote_command(ROOT, command)
    assert promoted[2] == "gleipnir.serving.stack.server"
    assert promoted[4] == "gleipnir.serving.stack.workers.gdn_fast.FastGdnWorker"
    additional = json.loads(promoted[6])
    assert additional["gleipnir_frost_fp4"] == {
        entry["path"]: entry["sha256"],
        "x.py": "1",
    }
    assert (
        additional["serving_condition"]["server_module"]
        == "gleipnir.serving.stack.registry"
    )
    assert promoted[8] == command[8]
    assert promoted_module("experiments.unmoved.thing") == "experiments.unmoved.thing"


def test_legacy_runtime_names_resolve_to_the_promoted_modules_cold():
    code = """
import importlib
stack = "gleipnir.serving.stack."
pairs = [
    ("experiments.b200_vllm031.runtime", stack + "environment"),
    ("experiments.b200_vllm031.source_bindings", stack + "source_bindings"),
    ("experiments.b200_context_million.envelope", stack + "long_context_envelope"),
]
for old, new in pairs:
    assert importlib.import_module(old) is importlib.import_module(new), old
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
    module = importlib.import_module("experiments.b200_context_million.envelope")
    assert module.__name__ == "gleipnir.serving.stack.long_context_envelope"


def test_the_serving_default_resolves_onto_the_promoted_stack():
    from gleipnir.serving.reference import selected_serving_default

    _, command = selected_serving_default(ROOT)
    modules = [a for a in command if a.startswith(("experiments.", "gleipnir."))]
    assert modules and all(m.startswith("gleipnir.serving.stack.") for m in modules)
    additional = json.loads(command[command.index("--additional-config") + 1])
    for path, digest in additional["gleipnir_frost_fp4"].items():
        assert not path.startswith(
            tuple(
                e.rsplit("/", 1)[0]
                for e in promotion()["sources"]
                if e.startswith("experiments/b200_gdn")
            )
        )
        if (ROOT / path).is_file() and path.startswith("src/gleipnir/serving/stack/"):
            assert _sha(ROOT / path) == digest


# Recorded sources that already differ on main 1bff5636 (archived canaries the
# launch restores, the pinned pre-PATH-fix __init__.py, and sources of older
# receipts reachable from the release stack). Anything else must still match.
KNOWN_UNMATCHED = {
    "experiments/b200_attention_gdn_serving/attention_fp4_canary.py",
    "experiments/b200_attention_gdn_serving/attention_fp4_worker.py",
    "experiments/b200_attention_gdn_serving/fp4_fusion_canary.py",
    "experiments/b200_attention_gdn_serving/fp4_gemm_tune.py",
    "experiments/b200_attention_gdn_serving/fp4_prepare_canary.py",
    "experiments/b200_attention_gdn_serving/fp4_swiglu_native_output_compare.py",
    "experiments/b200_attention_gdn_serving/fp4_swiglu_overhead_compare.py",
    "experiments/b200_attention_gdn_serving/mixed_worker.py",
    "experiments/b200_attention_gdn_serving/mxfp8_canary.py",
    "experiments/b200_attention_gdn_serving/mxfp8_worker.py",
    "experiments/b200_attention_gdn_serving/prepare_worker.py",
    "experiments/b200_attention_gdn_serving/swiglu_native_output_worker.py",
    "experiments/b200_attention_gdn_serving/tuned_worker.py",
    "experiments/b200_attention_gdn_serving/worker.py",
    "experiments/b200_attention_precision/run.py",
    "experiments/b200_attention_precision/test_precision.py",
    "experiments/b200_monitor_score/run.py",
    "experiments/b200_swiglu_output/native.py",
    "experiments/b200_swiglu_output/server.py",
    "experiments/b200_swiglu_output/test_output.py",
    "experiments/b200_swiglu_output/worker.py",
    "src/gleipnir/__init__.py",
    "src/gleipnir/native_fp4_training.py",
    "src/gleipnir/serving/monitor_score.py",
    "src/gleipnir/serving_compile_cache.py",
    "src/gleipnir/serving_mxfp8.py",
}


def test_release_reachable_receipts_bind_no_newly_changed_source():
    import re

    import pytest

    if not (ROOT / "results/qad-v02-optimized01/server.json").is_file():
        pytest.skip("archived release receipts are not available")
    code = [
        *(ROOT / "experiments/release_serving").rglob("*.py"),
        *(ROOT / "src/gleipnir/serving").rglob("*.py"),
        ROOT / "experiments/b200_inference_benchmark/serving_default.json",
        ROOT / "experiments/b200_inference_benchmark/baseline.json",
    ]
    pattern = re.compile(r"results/[\w./-]+\.json")
    receipts = {r for f in code for r in pattern.findall(f.read_text(errors="ignore"))}
    for receipt in list(receipts):
        path = ROOT / receipt
        if path.is_file():
            receipts |= set(pattern.findall(path.read_text(errors="ignore")))
    source = re.compile(r'"((?:experiments|src)/[\w/.-]+\.py)"\s*:\s*"([0-9a-f]{64})"')
    unmatched = set()
    for receipt in sorted(receipts):
        path = ROOT / receipt
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore").replace("/workspace/gleipnir/", "")
        for reference, digest in source.findall(text):
            if not recorded_source_matches(ROOT, reference, digest):
                unmatched.add(reference)
    assert unmatched <= KNOWN_UNMATCHED, sorted(unmatched - KNOWN_UNMATCHED)
