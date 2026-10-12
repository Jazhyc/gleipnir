"""Selected lean reference QAT with privileged Kimi targets joined by identity."""

import json
from contextlib import contextmanager
from pathlib import Path

import yaml

from gleipnir.data.monitoring import file_hash, write_json
from gleipnir.training import qat_reference, quantization_recovery
from gleipnir.training.distillation_targets import load_binary_targets

_MODEL = None


def load_teachers(workload, root: Path, output: Path, config):
    """Retain original Kimi provenance; the loss consumes only soft probabilities."""
    targets, receipt = load_binary_targets(
        workload,
        root / config["teacher_cache"],
        expected_sha256=config["teacher_sha256"],
        teacher_model="moonshotai/kimi-k3",
        prompt_contract="privileged",
    )
    if any(
        int(p >= 0.5) != row["label"] for p, row in zip(targets, workload, strict=True)
    ):
        raise ValueError("strict-filtered Kimi target/source agreement drift")
    write_json(output / "kimi_target_identity.json", receipt)
    return [{"soft_target": p} for p in targets], {
        "teacher_sha256": receipt["cache_sha256"],
        "teacher_model": receipt["teacher_model"],
        "prompt_contract": receipt["prompt_contract"],
    }


def verify_operator_reference(root: Path, output: Path) -> None:
    """Reuse unchanged decoder and arithmetic-preserving structural validation."""
    config = yaml.safe_load(
        (root / "experiments/kimi_direct_qat/config.yaml").read_text()
    )
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
