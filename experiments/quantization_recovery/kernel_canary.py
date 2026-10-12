"""Fresh bounded decode controls on the training runtime, before model probes."""

from pathlib import Path

import torch

from gleipnir.data.monitoring import file_hash, write_json
from gleipnir.kernels.fp4.cudnn_fp4_gemm import decode_operand, pack_operand
from gleipnir.kernels.fp4.decode import decode_bf16


def main():
    root = Path.cwd()
    generator = torch.Generator(device="cuda").manual_seed(0)
    checks = []
    for rows in (1, 17, 129):
        for width in (2560, 4096, 9216):
            x = torch.randn(
                rows, width, dtype=torch.bfloat16, device="cuda", generator=generator
            )
            x[0].zero_()
            operand = pack_operand(
                x, row_amax=True, chunked_rows=True, hardware_packing=True
            )
            expected = decode_operand(operand).to(torch.bfloat16)
            observed = decode_bf16(operand)
            passed = torch.equal(observed, expected)
            checks.append(
                {"rows": rows, "width": width, "weight": False, "passed": passed}
            )
            if not passed:
                raise ValueError("BF16 direct decoder activation parity failed")
    for rows, width in ((2560, 4096), (18432, 2560), (2560, 9216), (12288, 2560)):
        x = torch.randn(
            rows, width, dtype=torch.bfloat16, device="cuda", generator=generator
        )
        operand = pack_operand(x, weight=True)
        expected = decode_operand(operand).to(torch.bfloat16)
        observed = decode_bf16(operand)
        passed = torch.equal(observed, expected)
        checks.append({"rows": rows, "width": width, "weight": True, "passed": passed})
        if not passed:
            raise ValueError("BF16 direct decoder weight parity failed")
    write_json(
        root / "results/quantization-recovery01/decoder_canary.json",
        {
            "passed": True,
            "checks": checks,
            "source_sha256": file_hash(root / "src/gleipnir/kernels/fp4/decode.py"),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
        },
    )
    print("decoder_canary_passed", flush=True)


if __name__ == "__main__":
    main()
