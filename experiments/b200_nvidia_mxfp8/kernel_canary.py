"""Check actual NVIDIA MXFP8 arithmetic before permitting model training."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as functional
import yaml

from gleipnir.monitoring_systems_screen import sha256_file
from gleipnir.nvidia_mxfp8_attention import (
    BACKWARD_ENGINE,
    FORWARD_ENGINE,
    UPSTREAM_REVISION,
    mxfp8_attention,
    quantize,
)

ROOT = Path(__file__).resolve().parents[2]


def difference(actual: torch.Tensor, expected: torch.Tensor, limit: float) -> dict:
    """Retain absolute errors for analytically zero reference derivatives."""
    a, e = actual.float(), expected.float()
    norm = float(e.norm())
    error = float((a - e).norm())
    relative = error / norm if norm > 1e-7 else None
    finite = bool(torch.isfinite(a).all())
    return {
        "relative_l2": relative,
        "absolute_l2": error,
        "reference_l2": norm,
        "max_absolute": float((a - e).abs().max()),
        "limit": limit,
        "finite": finite,
        "passed": finite
        and (relative <= limit if relative is not None else error <= 1e-4),
    }


def main() -> None:
    import cudnn

    from gleipnir.flashqla_training import load_flashqla

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/b200_nvidia_mxfp8/config.yaml",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = ROOT / config["output"] / "kernel_canary.json"
    if output.exists():
        raise ValueError("native receipt already exists")
    report = {
        "status": "running",
        "passed": False,
        "upstream_revision": UPSTREAM_REVISION,
        "heads": config["heads"],
        "head_dim": config["head_dim"],
        "cases": [],
        "engines": [FORWARD_ENGINE, BACKWARD_ENGINE],
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cudnn_backend": cudnn.backend_version(),
        "torch_cudnn_backend": torch.backends.cudnn.version(),
        "cudnn_frontend": cudnn.__version__,
        "packages": {
            p: metadata.version(p)
            for p in ["nvidia-cutlass-dsl", "apache-tvm-ffi", "flash-attn-4"]
        },
        "cache_paths": {k: v for k, v in os.environ.items() if "CACHE" in k},
    }

    def publish() -> None:
        output.write_text(json.dumps(report, indent=2) + "\n")

    publish()
    try:
        _, report["recurrence"] = load_flashqla()
        source = Path(os.environ["GLEIPNIR_NVIDIA_SOURCE"])
        quantizer_path = source / "test/python/sdpa/mxfp8_quant.py"
        spec = importlib.util.spec_from_file_location(
            "nvidia_quantization_oracle", quantizer_path
        )
        oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
        report["quantization_oracle_sha256"] = sha256_file(quantizer_path)
        report["quantizer_layout"] = []
        torch.manual_seed(0)
        for length in config["lengths"]:
            print(f"native_case_start length={length}", flush=True)
            q, k, v = [
                torch.randn(
                    length, h, 256, device="cuda", dtype=torch.bfloat16
                ).requires_grad_()
                for h in config["heads"] + [config["heads"][1]]
            ]
            for h in config["heads"]:
                x = q.detach() if h == config["heads"][0] else k.detach()
                ref = oracle.quantize_to_mxfp8(
                    x.transpose(0, 1).unsqueeze(0), 1, h, length, 256, with_ref=False
                )
                for col, index in [(False, 0), (True, 3)]:
                    payload, scales = quantize(x, col)
                    expected_payload = ref[index][0].transpose(0, 1)
                    equal = torch.equal(
                        payload.view(torch.uint8),
                        expected_payload.contiguous().view(torch.uint8),
                    ) and torch.equal(scales, ref[index + 2].reshape(-1))
                    report["quantizer_layout"].append(
                        {
                            "length": length,
                            "heads": h,
                            "columnwise": col,
                            "passed": equal,
                        }
                    )
                    if not equal:
                        raise AssertionError(
                            "fused quantizer does not match upstream reference layout"
                        )
            rq, rk, rv = [x.detach().float().requires_grad_() for x in (q, k, v)]
            reference = functional.scaled_dot_product_attention(
                rq.transpose(0, 1),
                rk.transpose(0, 1),
                rv.transpose(0, 1),
                is_causal=True,
                enable_gqa=True,
            ).transpose(0, 1)
            candidate = mxfp8_attention(q, k, v)
            case = {
                "length": length,
                "status": "forward_complete",
                "errors": {
                    "forward": difference(
                        candidate, reference, config["forward_relative_l2_limit"]
                    )
                },
            }
            report["cases"].append(case)
            publish()
            gradient = torch.randn_like(candidate)
            reference.backward(gradient.float())
            candidate.backward(gradient)
            torch.cuda.synchronize()
            errors = {
                name: difference(actual, expected, limit)
                for name, actual, expected, limit in [
                    (
                        "forward",
                        candidate,
                        reference,
                        config["forward_relative_l2_limit"],
                    ),
                    ("dq", q.grad, rq.grad, config["gradient_relative_l2_limit"]),
                    ("dk", k.grad, rk.grad, config["gradient_relative_l2_limit"]),
                    ("dv", v.grad, rv.grad, config["gradient_relative_l2_limit"]),
                ]
            }
            case.update(
                status="complete",
                errors=errors,
                passed=all(e["passed"] for e in errors.values()),
            )
            publish()
            print(json.dumps(case), flush=True)

        q, k, v = [
            torch.randn(65, h, 256, device="cuda", dtype=torch.bfloat16)
            for h in [16, 4, 4]
        ]
        original = mxfp8_attention(q, k, v)
        within = v.clone()
        within[16:32] *= 8
        across = v.clone()
        across[32:] *= 8
        within_result = mxfp8_attention(q, k, within)
        across_result = mxfp8_attention(q, k, across)
        report["causal_quantization"] = {
            "within_block_future_perturbation_max_absolute": float(
                (original[:16] - within_result[:16]).abs().max()
            ),
            "across_block_future_perturbation_max_absolute": float(
                (original[:32] - across_result[:32]).abs().max()
            ),
            "block_size": 32,
            "note": (
                "Columnwise V quantization may couple earlier values to future "
                "values within a scale block; reported separately from masking."
            ),
        }
        if (
            report["causal_quantization"][
                "across_block_future_perturbation_max_absolute"
            ]
            != 0
        ):
            raise AssertionError("causal mask or quantizer leaks across scale blocks")
        report.update(
            status="complete", passed=all(c["passed"] for c in report["cases"])
        )
        publish()
        if not report["passed"]:
            raise AssertionError(
                "native MXFP8 numerical gate failed; no model updates permitted"
            )
    except Exception as error:
        report.update(
            status="failed", passed=False, error=f"{type(error).__name__}: {error}"
        )
        publish()
        raise


if __name__ == "__main__":
    main()
