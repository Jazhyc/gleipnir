"""Check native FP4 arithmetic on the remaining decoder projection shapes."""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
from pathlib import Path

from experiments.local_inference.core import write_json
from gleipnir.qwen35_adapter_rebase import sha256_file

SHAPES = ((12288, 2560), (64, 2560), (2560, 4096), (10240, 2560))


def validate_shapes(summary: dict) -> None:
    """Require every tested shape to occur in the actual decoder profile."""
    if summary["unmatched_gemm_seconds"] != 0:
        raise ValueError("Unaccounted profile matrix operations")
    observed = {
        (r["n"], r["k"])
        for r in summary["shapes"]
        if "attn" in r["projection"] or "attention" in r["projection"]
    }
    if observed != set(SHAPES):
        raise ValueError("Decoder shape binding differs from the frozen screen")


def main() -> None:
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
    )

    from experiments.fp4_inference.kernel_canary import decode, relative_error
    from gleipnir.vllm_nvfp4 import NVFP4_MAX, native_linear, pack_weight

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("results/fp4_inference/profile_bf16_shapes/shape_summary.json"),
    )
    args = parser.parse_args()
    summary = json.loads(args.profile.read_text())
    validate_shapes(summary)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(20260929)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    sources = (
        Path("src/gleipnir/__init__.py"),
        Path("src/gleipnir/_compat.py"),
        Path(__file__),
        Path("src/gleipnir/serving/vllm/nvfp4.py"),
        Path("src/gleipnir/kernels/fp4/nvfp4_reference.py"),
        Path("experiments/fp4_inference/kernel_canary.py"),
    )
    result = {
        "state": "running",
        "utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "seed": 20260929,
        "profile_sha256": sha256_file(args.profile),
        "sources": {str(p): sha256_file(p) for p in sources},
        "device": torch.cuda.get_device_name(),
        "software": {
            p: importlib.metadata.version(p)
            for p in ("torch", "vllm", "flashinfer-python")
        },
        "backend": "cutlass",
        "activation_scale": "dynamic tensor-global; group16 E4M3",
        "maximum_relative_l2": 0.01,
        "checks": [],
        "note": (
            "Synthetic normal BF16 values at observed model shapes; independent "
            "decoded FP32 reference. Arithmetic support only; "
            "no quality or speed claim."
        ),
    }
    receipt = args.output / "result.json"
    write_json(receipt, result)
    try:
        with torch.inference_mode():
            for n, k in SHAPES:
                weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
                packed, blocks, scale = pack_weight(weight)
                swizzled = swizzle_blockscale(blocks)
                decoded_weight = decode(packed, blocks, scale)
                for m in (1, 128, 2048):
                    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
                    xs = x.abs().amax().float().clamp_min(1e-12).reshape(1) / NVFP4_MAX
                    xp, xb = ops.scaled_fp4_quant(x, xs.reciprocal(), False)
                    reference = decode(xp, xb, xs) @ decoded_weight.t()
                    output = native_linear(x, packed, swizzled, scale, "cutlass")
                    error = relative_error(output, reference)
                    passed = bool(torch.isfinite(output).all()) and error <= 0.01
                    check = {
                        "shape_mnk": [m, n, k],
                        "relative_l2": error,
                        "passed": passed,
                    }
                    result["checks"].append(check)
                    write_json(receipt, result)
                    print(json.dumps(check), flush=True)
                    if not passed:
                        raise ValueError("Native decoder-shape arithmetic gate failed")
                del weight, packed, blocks, swizzled, decoded_weight
        result["state"] = "complete"
        write_json(receipt, result)
    except Exception as error:
        result.update(state="failed", error=str(error))
        write_json(receipt, result)
        raise


if __name__ == "__main__":
    main()
