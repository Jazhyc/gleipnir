"""Lean original QAD arithmetic with source-bound startup receipt reuse."""

import json
from contextlib import contextmanager
from pathlib import Path

from gleipnir.data.monitoring import file_hash, write_json
from gleipnir.training import qat_reference, quantization_recovery

_MODEL = None


def verify_operator_reference(root: Path, output: Path) -> None:
    config = json.loads((output / "execution_contract.json").read_text())["config"]
    for source, expected in config["reference_sources"].items():
        if file_hash(root / source) != expected:
            raise ValueError(f"validated source drift: {source}")
    receipts = {}
    for key, flag in (
        ("structural_reference", "numerical_passed"),
        ("decoder_reference", "passed"),
        ("startup_reference", "passed"),
    ):
        spec = config[key]
        path = root / spec["path"]
        if file_hash(path) != spec["sha256"] or not json.loads(path.read_text())[flag]:
            raise ValueError(f"startup reference drift: {key}")
        receipts[key] = {"reused": True, **spec}
    startup = json.loads((root / config["startup_reference"]["path"]).read_text())
    if startup["initial_master_sha256"] != config["initial_adapter_sha256"]:
        raise ValueError("reused startup initialization drift")
    write_json(output / "startup_reuse.json", receipts)


def install(model, root: Path):
    global _MODEL
    result = quantization_recovery.install(model, root)
    _MODEL = model
    return {
        **result,
        "structural_optimization": {"mlp": True, "gdn": True},
        "attention_backward": "quantized_QKV_BF16_FA4_surrogate",
    }


@contextmanager
def attention_runtime():
    if _MODEL is None:
        raise RuntimeError("install merged QAD projections before runtime")
    with qat_reference.attention_runtime(_MODEL):
        yield
