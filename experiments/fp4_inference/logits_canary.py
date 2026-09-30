"""Validate FP32 output matmul against independent FP32 reference on real inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.fp4_inference.kernel_canary import checked_load, relative_error
from experiments.local_inference.core import write_json
from gleipnir.qwen35_adapter_rebase import sha256_file
from gleipnir.vllm_fp32_logits import Fp32LogitsMethod


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path("results/fp4_inference/capture")
    manifest = json.loads((root / "manifest.json").read_text())
    weights = checked_load(root / "weights.pt", manifest["weights_sha256"])
    row = checked_load(root / "row_0.pt", manifest["completed"][0]["sha256"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    results = []
    for key in ("0_gate_up", "0_down"):
        x, weight = row[key][:16].cuda(), weights[key].cuda()
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        native = Fp32LogitsMethod().apply(layer, x)
        expected = x.float() @ weight.float().t()
        error = relative_error(native, expected)
        if (
            native.dtype != torch.float32
            or not torch.isfinite(native).all().item()
            or error > 0.001
        ):
            raise ValueError("FP32 output projection arithmetic failure")
        results.append(
            {"projection": key, "relative_l2": error, "output_dtype": str(native.dtype)}
        )
    write_json(
        args.output / "result.json",
        {
            "state": "complete",
            "results": results,
            "source_sha256": sha256_file(Path(__file__)),
            "method_sha256": sha256_file(Path("src/gleipnir/vllm_fp32_logits.py")),
        },
    )
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
