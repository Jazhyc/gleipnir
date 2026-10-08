"""Compare paired packing, QK, BF16 GEMM and normalization across Torch versions."""

import argparse
import importlib.metadata
from pathlib import Path

import torch

from experiments.b200_inference_benchmark.run import write
from gleipnir.flashqla_training import tensor_comparison


def normalize(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    summed = x.float() + residual.float()
    variance = summed.square().mean(dim=-1, keepdim=True)
    return (summed * torch.rsqrt(variance + 1e-6) * (weight.float() + 1)).to(x.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("generate", "evaluate", "compare"), required=True
    )
    parser.add_argument("--label", choices=("old", "new"))
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory
    if args.mode == "generate":
        directory.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(310)
        cases = []
        for rows in (17, 129, 270):
            cases.append(
                {
                    "x": torch.randn(rows, 2560, dtype=torch.bfloat16),
                    "r": torch.randn(rows, 2560, dtype=torch.bfloat16),
                    "w": torch.randn(2560, dtype=torch.bfloat16) * 0.01,
                    "linear_w": torch.randn(64, 2560, dtype=torch.bfloat16) * 0.02,
                    "q": torch.randn(rows, 8192, dtype=torch.bfloat16),
                    "k": torch.randn(rows, 1024, dtype=torch.bfloat16),
                    "qw": torch.randn(256, dtype=torch.bfloat16) * 0.01,
                    "kw": torch.randn(256, dtype=torch.bfloat16) * 0.01,
                    "rope": torch.randn(4096, 64, dtype=torch.bfloat16),
                    "pos": torch.arange(rows),
                }
            )
        torch.save(cases, directory / "inputs.pt")
        return
    if args.mode == "compare":
        a = torch.load(directory / "old.pt", weights_only=True)
        b = torch.load(directory / "new.pt", weights_only=True)
        checks = []
        for old, new in zip(a, b, strict=True):
            checks.append(
                {
                    key: {
                        **tensor_comparison(new[key], old[key]),
                        "exact": torch.equal(new[key], old[key]),
                    }
                    for key in old
                }
            )
        write(directory / "comparison.json", {"checks": checks})
        print(checks, flush=True)
        return
    if args.label is None:
        raise ValueError("evaluation requires the runtime label")
    from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate

    from gleipnir.cudnn_fp4_gemm import decode_operand
    from gleipnir.serving.fp4.prepare import vendor_pack

    compiled = torch.compile(normalize, fullgraph=True, dynamic=True)
    outputs = []
    for cpu in torch.load(directory / "inputs.pt", weights_only=True):
        c = {k: v.cuda() for k, v in cpu.items()}
        packed = vendor_pack(c["x"])
        qk = fused_qk_rmsnorm_rope_gate(
            c["q"],
            c["k"],
            c["qw"].float() + 1,
            c["kw"].float() + 1,
            c["rope"],
            c["pos"],
            1e-6,
            16,
            4,
            256,
            64,
        )
        outputs.append(
            {
                "norm_eager": normalize(c["x"], c["r"], c["w"]).cpu(),
                "norm_compiled": compiled(c["x"], c["r"], c["w"]).cpu(),
                "bf16_linear": torch.nn.functional.linear(c["x"], c["linear_w"]).cpu(),
                "packed_decoded": decode_operand(packed).cpu(),
                "packed_inverse": packed.inverse.cpu(),
                "q": qk[0].cpu(),
                "k": qk[1].cpu(),
                "gate": qk[2].cpu(),
            }
        )
    torch.save(outputs, directory / f"{args.label}.pt")
    write(
        directory / f"{args.label}_runtime.json",
        {
            key: importlib.metadata.version(key)
            for key in ("torch", "triton", "vllm", "flashinfer-python")
        },
    )
    print("runtime_native_complete", args.label, flush=True)


if __name__ == "__main__":
    main()
