"""Check fused device predicates and the declared within-block coupling."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from gleipnir.nvidia_mxfp8_fused_quantize import _boundaries, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve existing audit receipts")
    cases = [
        ((0, 1, 2), 2, 1, True),
        (tuple(range(33)), 32, 1, True),
        ((0, 1, 1), 1, 1, False),
        ((0, 2, 1), 1, 2, False),
        ((1, 2), 2, 1, False),
        ((-1, 2), 2, 3, False),
        ((0, 2), 3, 2, False),
        ((0, 2), 2, 1, False),
    ]
    report = {
        "boundary_cases": [],
        "coupling": [],
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path("src/gleipnir/nvidia_mxfp8_fused_quantize.py"),
            ]
        },
    }
    for offsets, total, maximum, expected in cases:
        cuts = torch.tensor(offsets, dtype=torch.int32, device="cuda")
        flag = torch.empty((), device="cuda", dtype=torch.bool)
        # Observe predicates directly so negative cases do not poison the CUDA
        # context with the final asynchronous assertion used in training.
        _boundaries[(1,)](cuts, flag, total, maximum, len(offsets) - 1, num_warps=1)
        actual = bool(flag)
        report["boundary_cases"].append(
            {"offsets": offsets, "expected": expected, "actual": actual}
        )
        if actual is not expected:
            raise RuntimeError("boundary predicate mismatch")
    cuts = torch.tensor([0, 65], dtype=torch.int32, device="cuda")
    for square in (False, True):
        x = torch.ones(65, 4, 256, device="cuda", dtype=torch.bfloat16)
        baseline = prepare(x, cuts, 65, square=square)
        x[32].mul_(1024)
        across = prepare(x, cuts, 65, square=square)
        x[31].mul_(2048)
        within = prepare(x, cuts, 65, square=square)
        unaffected = all(
            torch.equal(a[0].view(torch.uint8), b[0].view(torch.uint8))
            for a, b in zip(baseline[:2], across[:2], strict=True)
        )
        changed = [
            not torch.equal(a[0].view(torch.uint8), b[0].view(torch.uint8))
            for a, b in zip(baseline[:2], within[:2], strict=True)
        ]
        report["coupling"].append(
            {
                "square": square,
                "across_32_token_boundary_unchanged": unaffected,
                "within_block_changes_row_column_payload": changed,
            }
        )
        if not unaffected or changed != ([True, True] if square else [False, True]):
            raise RuntimeError("quantization block scope differs from the contract")
    report["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
