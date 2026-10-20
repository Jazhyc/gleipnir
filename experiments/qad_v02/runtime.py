"""Selected lean reference QAT arithmetic for the Gleipnir 4B v0.2 recovery.

Identical to the Kimi-direct runtime hooks except for its config and the
absence of Kimi targets: the loss consumes the v0.2 BF16 teacher cache.
"""

import json
from contextlib import contextmanager
from pathlib import Path

import yaml

from gleipnir.data.monitoring import file_hash, write_json
from gleipnir.training import qat_reference, quantization_recovery

CONFIG = "experiments/qad_v02/config.yaml"
_MODEL = None


def verify_operator_reference(root: Path, output: Path) -> None:
    """Reuse the unchanged decoder and structural validation, source-bound."""
    config = yaml.safe_load((root / CONFIG).read_text())
    for source, expected in config["reference_sources"].items():
        if file_hash(root / source) != expected:
            raise ValueError(f"validated QAT source drift: {source}")
    receipts = {}
    for key, flag in (
        ("structural_reference", "numerical_passed"),
        ("decoder_reference", "passed"),
    ):
        spec = config[key]
        path = root / spec["path"]
        if file_hash(path) != spec["sha256"] or not json.loads(path.read_text())[flag]:
            raise ValueError(f"QAT reference receipt drift: {key}")
        receipts[key] = {"reused": True, **spec}
    write_json(output / "operator_reference.json", receipts)


def install(model, root: Path):
    global _MODEL
    result = quantization_recovery.install(model, root)
    _MODEL = model
    return {
        **result,
        "projection_gemm": "decoded_FP8_FP32_reference",
        "structural_optimization": {"mlp": True, "gdn": True},
        "attention_backward": "quantized_QKV_BF16_FA4_surrogate",
    }


@contextmanager
def attention_runtime():
    if _MODEL is None:
        raise RuntimeError("install QAT projections before entering the runtime")
    with qat_reference.attention_runtime(_MODEL):
        yield
