"""Populate FLA autotune caches before the long-context model is resident."""

from __future__ import annotations

import argparse
import json

import torch


def prewarm_gated_delta_rule(sequence_length: int) -> dict[str, int | str]:
    """Run the Qwen3.5 gated-delta forward/backward at an exact sequence length."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("FLA prewarm requires CUDA")

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tensor_shape = (1, sequence_length, 32, 128)
    gate_shape = (1, sequence_length, 32)
    query = torch.randn(tensor_shape, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(tensor_shape, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(tensor_shape, device=device, dtype=dtype, requires_grad=True)
    gate = -torch.rand(gate_shape, device=device, dtype=torch.float32)
    gate.requires_grad_(True)
    beta = torch.rand(gate_shape, device=device, dtype=dtype, requires_grad=True)

    torch.cuda.reset_peak_memory_stats(device)
    output, _ = chunk_gated_delta_rule(
        query,
        key,
        value,
        g=gate,
        beta=beta,
        use_qk_l2norm_in_kernel=True,
    )
    output[..., 0, 0].float().mean().backward()
    torch.cuda.synchronize(device)
    result: dict[str, int | str] = {
        "device": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "sequence_length": sequence_length,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "status": "passed",
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, required=True)
    args = parser.parse_args()
    prewarm_gated_delta_rule(args.sequence_length)


if __name__ == "__main__":
    main()
