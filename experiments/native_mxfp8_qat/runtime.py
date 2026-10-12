"""Change only attention backward in the completed native-FP8 QAT recipe."""

import json
from contextlib import contextmanager
from pathlib import Path

import torch
import yaml

from experiments.native_fp8_qat import runtime as control
from experiments.quantization_recovery.train import binary_kl
from gleipnir.data.monitoring import file_hash, write_json
from gleipnir.training import mxfp8_backward

__all__ = ["attention_runtime", "install", "verify_operator_reference"]


def verify_operator_reference(root: Path, output: Path) -> None:
    """Bind the unchanged operator checks and isolated native-backward receipt."""
    control.verify_operator_reference(root, output)
    config = yaml.safe_load(
        (root / "experiments/native_mxfp8_qat/config.yaml").read_text()
    )
    spec = config["native_backward_reference"]
    path = root / spec["path"]
    if file_hash(path) != spec["sha256"]:
        raise ValueError("native backward probe receipt drift")
    receipt = json.loads(path.read_text())
    if receipt["status"] != "complete":
        raise ValueError("native backward probe incomplete")
    for source, checksum in receipt["sources"].items():
        if file_hash(root / source) != checksum:
            raise ValueError("native backward validated-source drift")
    write_json(output / "backward_reference.json", {"reused": True, **spec})


def install(model, root: Path):
    """Keep native FP8 projections and accepted structural optimizations fixed."""
    return {
        **control.install(model, root),
        "attention_backward": "native_MXFP8_matching_serving_forward_LSE",
    }


@contextmanager
def attention_runtime():
    """The inner registry overrides only the control's attention backend."""
    with control.attention_runtime(), mxfp8_backward.attention_runtime():
        yield


def validate_model_backward(model, logits, rows, output: Path) -> None:
    """Compare complete initial-adapter gradients without mutating training state."""
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    mode = model.training
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    norms, differences, snapshots = {}, {}, {}
    original_logits = None
    try:
        model.train()
        for backend, scope in (
            ("fa4_surrogate", control.attention_runtime),
            ("native_mxfp8", attention_runtime),
        ):
            torch.set_rng_state(cpu_rng)
            if cuda_rng:
                torch.cuda.set_rng_state_all(cuda_rng)
            model.zero_grad(set_to_none=True)
            with scope():
                values = logits(rows)
                targets = torch.full((len(rows),), 0.5, device=values.device)
                loss = binary_kl(values, targets).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite paired whole-model loss")
                loss.backward()
            if any(
                p.grad is None
                or not torch.isfinite(p.grad).all()
                or p.grad.dtype != torch.float32
                or p.dtype != torch.float32
                for p in parameters.values()
            ):
                raise FloatingPointError("missing/nonfinite/non-FP32 paired gradients")
            norms[backend] = sum(
                float(p.grad.double().square().sum()) for p in parameters.values()
            )
            if original_logits is None:
                original_logits = values.detach().clone()
                snapshots = {
                    n: p.grad.detach().cpu().clone() for n, p in parameters.items()
                }
            else:
                difference = float((values.detach() - original_logits).abs().max())
                for name, parameter in parameters.items():
                    baseline = snapshots[name].to(parameter.device)
                    delta = parameter.grad.double() - baseline.double()
                    squared = float(delta.square().sum())
                    reference_squared = float(baseline.double().square().sum())
                    differences[name] = {
                        "difference_squared": squared,
                        "reference_squared": reference_squared,
                        "relative_l2": (squared / max(reference_squared, 1e-24)) ** 0.5,
                    }
                receipt = {
                    "passed": difference == 0,
                    "scope": "two_short_training_rows_all_initial_adapter_gradients",
                    "ids": [row["id"] for row in rows],
                    "forward_maximum_absolute_difference": difference,
                    "gradient_relative_l2": (
                        sum(item["difference_squared"] for item in differences.values())
                        / max(norms["fa4_surrogate"], 1e-24)
                    )
                    ** 0.5,
                    "gradient_norms": {
                        name: value**0.5 for name, value in norms.items()
                    },
                    "per_tensor": differences,
                    "gradient_reproduction_required": False,
                    "promotion": False,
                }
                write_json(output / "whole_model_backward.json", receipt)
                print(
                    "whole_model_backward",
                    json.dumps({k: v for k, v in receipt.items() if k != "per_tensor"}),
                    flush=True,
                )
                if not receipt["passed"]:
                    raise ValueError(
                        "native backward pairing changed whole-model forward"
                    )
    finally:
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
        model.train(mode)
