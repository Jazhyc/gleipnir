"""Select the explicit new native FP8 variant for the frozen training template."""

import json
from contextlib import contextmanager
from pathlib import Path

import yaml

from gleipnir.data.monitoring import file_hash, write_json
from gleipnir.training import quantization_recovery as reference
from gleipnir.training.qat_optimization import optimized_path

_MODEL = None


def verify_operator_reference(root: Path, output: Path) -> None:
    """Reuse unchanged native FP8/FP4 decoder receipts with exact source binding."""
    config = yaml.safe_load(
        (root / "experiments/native_fp8_qat/config.yaml").read_text()
    )
    spec = config["native_operator_reference"]
    path = root / spec["path"]
    if file_hash(path) != spec["sha256"] or not json.loads(path.read_text())["passed"]:
        raise ValueError("native FP8 operator reference drift")
    for path, expected in spec["sources"].items():
        if file_hash(root / path) != expected:
            raise ValueError(f"native operator source drift: {path}")
    decoder = spec["decoder"]
    decoder_path = root / decoder["path"]
    if file_hash(decoder_path) != decoder["sha256"]:
        raise ValueError("FP4 decoder reference drift")
    decoder_receipt = json.loads(decoder_path.read_text())
    if not decoder_receipt["passed"] or decoder_receipt["source_sha256"] != file_hash(
        root / "src/gleipnir/kernels/fp4/decode.py"
    ):
        raise ValueError("FP4 decoder source/validation drift")
    write_json(output / "operator_reference.json", {"reused": True, **spec})


def install(model, root: Path):
    global _MODEL
    result = reference.install(model, root)
    _MODEL = model
    return {
        **result,
        "projection_gemm": "native_FP8_FP32_accumulation",
        "structural_optimization": {"mlp": True, "gdn": True},
        "attention_backward": "quantized_QKV_FA4_surrogate",
    }


@contextmanager
def attention_runtime():
    if _MODEL is None:
        raise RuntimeError("install QAT projections before choosing the runtime")
    with (
        reference.attention_runtime(),
        optimized_path(_MODEL, fp8=True, mlp=True, gdn=True),
    ):
        yield
