"""Execute fused preparation layout, arithmetic and packing diagnostics."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import traceback
from itertools import accumulate
from pathlib import Path
from unittest.mock import patch

import torch

import experiments.b200_nvidia_mxfp8_varlen.kernel_canary as checks
from experiments.b200_mxfp8_fused.square_reference import square_reference
from gleipnir.nvidia_mxfp8_fused_attention import packed_attention
from gleipnir.nvidia_mxfp8_fused_quantize import prepare
from gleipnir.nvidia_mxfp8_varlen_attention import packed_attention as old_attention
from gleipnir.nvidia_mxfp8_varlen_quantize import quantize
from gleipnir.nvidia_mxfp8_varlen_repack import repack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--square", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "starting",
        "square": args.square,
        "quantizer_checks": [],
        "cache_paths": {k: v for k, v in os.environ.items() if "CACHE" in k},
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                *Path("src/gleipnir").glob("nvidia_mxfp8_fused*.py"),
            ]
        },
    }
    for name in report["source_sha256"]:
        path = Path(name)
        relative = path.relative_to(Path.cwd()) if path.is_absolute() else path
        dest = args.output / "executed_source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())

    def save():
        (args.output / "kernel_canary.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in {"cache_paths", "source_sha256"}
                }
            ),
            flush=True,
        )

    save()
    try:
        torch.manual_seed(17)
        lengths = (1, 3, 31, 33, 63, 65, 127, 129, 255, 257)
        offsets = (0, *accumulate(lengths))
        total, maximum = sum(lengths), max(lengths)
        cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
        for heads in (16, 4):
            x = torch.randn(total, heads, 256, device="cuda", dtype=torch.bfloat16)
            x *= (2.0 ** (torch.arange(total, device="cuda") % 16 - 8))[:, None, None]
            x *= (2.0 ** (torch.arange(256, device="cuda") // 32 - 3))[None, None, :]
            fused = prepare(x, cu, maximum, square=args.square)
            if not args.square:
                row, row_sf, packed_row = quantize(x, cu, maximum, False)
                col, col_sf, packed_col = quantize(x, cu, maximum, True)
                expected = (
                    row,
                    col,
                    repack(
                        row_sf,
                        maximum,
                        len(lengths),
                        heads,
                        cu,
                        columnwise=False,
                        sfa=True,
                    ),
                    repack(
                        row_sf,
                        maximum,
                        len(lengths),
                        heads,
                        cu,
                        columnwise=False,
                        sfa=False,
                    ),
                    repack(
                        col_sf,
                        maximum,
                        len(lengths),
                        heads,
                        cu,
                        columnwise=True,
                        sfa=False,
                    ),
                )
                equality = [
                    torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                    for a, b in zip(fused[:5], expected, strict=True)
                ]
                # Only actual forward scale tiles are read; unused capacity is dead.
                capacity = (total + 127) // 128 + len(lengths)
                live = sum((n + 127) // 128 for n in lengths)
                equality += [
                    torch.equal(
                        a.view(heads, capacity, 1024)[:, :live],
                        b.view(heads, capacity, 1024)[:, :live],
                    )
                    for a, b in zip(fused[5:], (packed_row, packed_col), strict=True)
                ]
                report["quantizer_checks"].append(
                    {"heads": heads, "seven_layouts_bitexact": equality}
                )
                save()
                if not all(equality):
                    raise RuntimeError(
                        "fused preparation differs from pinned arithmetic/layout"
                    )
            else:
                payload, row_sf, col_sf, packed_row, packed_col = square_reference(
                    x, lengths, maximum
                )
                expected = (
                    payload,
                    payload,
                    repack(
                        row_sf,
                        maximum,
                        len(lengths),
                        heads,
                        cu,
                        columnwise=False,
                        sfa=True,
                    ),
                    repack(
                        row_sf,
                        maximum,
                        len(lengths),
                        heads,
                        cu,
                        columnwise=False,
                        sfa=False,
                    ),
                    repack(
                        col_sf,
                        maximum,
                        len(lengths),
                        heads,
                        cu,
                        columnwise=True,
                        sfa=False,
                    ),
                )
                equality = [
                    torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                    for a, b in zip(fused[:5], expected, strict=True)
                ]
                capacity = (total + 127) // 128 + len(lengths)
                live = sum((n + 127) // 128 for n in lengths)
                equality += [
                    torch.equal(a.view(heads, capacity, 1024)[:, :live], b)
                    for a, b in zip(fused[5:], (packed_row, packed_col), strict=True)
                ]
                report["quantizer_checks"].append(
                    {
                        "heads": heads,
                        "shared_payload": fused[0].data_ptr() == fused[1].data_ptr(),
                        "seven_reference_layouts_bitexact": equality,
                    }
                )
                if fused[0].data_ptr() != fused[1].data_ptr():
                    raise RuntimeError("square payload was not shared")
                if not all(equality):
                    raise RuntimeError(
                        "square preparation differs from independent block reference"
                    )
        function = functools.partial(packed_attention, square=args.square)
        q, k, v = [
            torch.randn(
                total,
                heads,
                256,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for heads in (16, 4, 4)
        ]
        grad = torch.randn_like(q)
        out = function(q, k, v, cu, maximum)
        gradients = torch.autograd.grad(out, (q, k, v), grad)
        for x, derivative in zip((q, k, v), gradients, strict=True):
            x.grad = derivative
        report["finite"] = bool(
            torch.isfinite(out).all()
            and all(torch.isfinite(g).all() for g in gradients)
        )
        if not report["finite"]:
            raise RuntimeError("nonfinite attention/gradient")
        if not args.square:
            reference = old_attention(q, k, v, cu, maximum)
            derivatives = torch.autograd.grad(reference, (q, k, v), grad)
            errors = [
                checks.error(a, b) for a, b in zip(gradients, derivatives, strict=True)
            ]
            report["old_native_comparison"] = {
                "forward_relative_l2": checks.error(out, reference),
                "gradient_relative_l2": errors,
            }
            if report["old_native_comparison"]["forward_relative_l2"] or any(errors):
                raise RuntimeError("fused dual arithmetic changed")
        reference, derivatives = checks.check_fp32(q, k, v, grad, offsets)
        errors = [
            checks.error(a, b) for a, b in zip(gradients, derivatives, strict=True)
        ]
        forward_error = checks.error(out, reference)
        report["fp32_comparison"] = {
            "forward_relative_l2": forward_error,
            "gradient_relative_l2": errors,
            "strict_passed": forward_error <= 0.02 and max(errors) <= 0.05,
        }

        def segmented(q, k, v, offsets):
            return torch.cat(
                [
                    function(
                        q[a:b],
                        k[a:b],
                        v[a:b],
                        torch.tensor([0, b - a], device="cuda", dtype=torch.int32),
                        b - a,
                    )
                    for a, b in zip(offsets[:-1], offsets[1:], strict=True)
                ]
            )

        with (
            patch.object(checks, "packed_attention", function),
            patch.object(checks, "dense_reference", segmented),
        ):
            report["isolation"] = checks.check_isolation(q, k, v, cu, maximum, offsets)
            report["graph_replay"] = checks.check_replay(
                q, k, v, cu, maximum, grad, offsets
            )
        save()
        if not report["isolation"]["passed"] or not report["graph_replay"]["passed"]:
            raise RuntimeError("packing isolation or changed-cut graph replay failed")
        original_empty = torch.empty

        def poisoned_empty(*a, **kw):
            result = original_empty(*a, **kw)
            if result.dtype == torch.uint8:
                result.fill_(255)
            elif result.is_floating_point():
                result.fill_(float("nan"))
            return result

        with patch("torch.empty", poisoned_empty):
            poisoned = function(q, k, v, cu, maximum)
            poisoned_grad = torch.autograd.grad(poisoned, (q, k, v), grad)
        report["poisoned_dead_storage"] = {
            "forward_relative_l2": checks.error(poisoned, out),
            "gradient_relative_l2": [
                checks.error(a, b)
                for a, b in zip(poisoned_grad, gradients, strict=True)
            ],
        }
        if report["poisoned_dead_storage"]["forward_relative_l2"] or any(
            report["poisoned_dead_storage"]["gradient_relative_l2"]
        ):
            raise RuntimeError("poisoned allocation changes arithmetic")
        report["status"] = "execution_complete"
        save()
    except Exception:
        report["status"] = "failed"
        report["exception"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    main()
