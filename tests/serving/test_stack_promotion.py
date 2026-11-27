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
