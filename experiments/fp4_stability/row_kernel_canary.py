"""Require bitwise scaling parity before using fused FP4 row kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gleipnir.fouroversix_training import normalize_activation_rows
from gleipnir.fp4_row_kernels import normalize_rows, rescale_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    cases = []
    for rows, columns in [
        (1, 256),
        (128, 2560),
        (513, 9216),
        (16384, 2560),
        (256, 9216),
    ]:
        values = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)
        values[0] = 0
        if rows > 1:
            values[1] *= 1024
        expected, expected_scales = normalize_activation_rows(values)
        actual, scales = normalize_rows(values)
        if not torch.equal(actual, expected) or not torch.equal(
            scales, expected_scales
        ):
            raise ValueError(
                f"fused normalization is not bitwise equal at {(rows, columns)}"
            )
        result = torch.randn_like(values)
        expected_output = (result.float() * scales).to(torch.bfloat16)
        output = rescale_rows(result, scales)
        if not torch.equal(output, expected_output):
            raise ValueError(
                f"fused rescaling is not bitwise equal at {(rows, columns)}"
            )

        def measure(action):
            for _ in range(5):
                action()
            start, stop = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(20):
                action()
            stop.record()
            stop.synchronize()
            return start.elapsed_time(stop) / 20

        cases.append(
            {
                "shape": [rows, columns],
                "bitwise_normalization": True,
                "bitwise_rescaling": True,
                "unfused_normalization_ms": measure(
                    lambda values=values: normalize_activation_rows(values)
                ),
                "fused_normalization_ms": measure(
                    lambda values=values: normalize_rows(values)
                ),
                "unfused_rescaling_ms": measure(
                    lambda result=result, scales=scales: (result.float() * scales).to(
                        torch.bfloat16
                    )
                ),
                "fused_rescaling_ms": measure(
                    lambda result=result, scales=scales: rescale_rows(result, scales)
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"status": "passed", "cases": cases}, indent=2) + "\n"
    )
    print(json.dumps(cases), flush=True)


if __name__ == "__main__":
    main()
