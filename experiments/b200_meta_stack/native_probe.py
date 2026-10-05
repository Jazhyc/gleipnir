"""Run one native candidate, with fresh execution checks and bounded timing."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys
import traceback
from itertools import accumulate
from pathlib import Path

import torch

from gleipnir.nvidia_mxfp8_meta_variants import NativeVariant, native_variant

VARIANTS = {
    "baseline": NativeVariant(),
    "persistent_dq": NativeVariant(persistent_dq=True),
    "persistent_dkdv": NativeVariant(persistent_dkdv=True),
    "persistent_both": NativeVariant(persistent_dq=True, persistent_dkdv=True),
    "ds_warp_amax": NativeVariant(ds_warp_amax=True),
    "persistent_ds_warp_amax": NativeVariant(
        persistent_dq=True, persistent_dkdv=True, ds_warp_amax=True
    ),
    "dq_store128": NativeVariant(dq_store_bits=128),
    "p_scale16": NativeVariant(forward_p_scale_log2=4),
    "p_scale256": NativeVariant(forward_p_scale_log2=8),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "variant": args.variant, "rows": []}
    paths = [Path(__file__), *Path("src/gleipnir").glob("nvidia_mxfp8*.py")]
    report["source_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    for p in paths:
        destination = args.output / "executed_source" / p
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(p.read_bytes())

    def save():
        (args.output / "probe.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        with native_variant(VARIANTS[args.variant]) as spec:
            report["spec"] = spec
            from experiments.b200_mxfp8_fused import kernel_canary
            from experiments.b200_mxfp8_fused.training_screen import accept_native
            from experiments.b200_nvidia_mxfp8_varlen.profile_attention import (
                SHAPES,
                timing,
            )
            from gleipnir.nvidia_mxfp8_fused_attention import packed_attention

            sys.argv = [
                "kernel_canary",
                "--output",
                str(args.output / "canary"),
                "--square",
            ]
            kernel_canary.main()
            receipt = args.output / "canary/kernel_canary.json"
            canary = json.loads(receipt.read_text())
            report["canary_sha256"] = hashlib.sha256(receipt.read_bytes()).hexdigest()
            report["native_acceptance"] = accept_native(canary, square=True)
            save()
            torch.manual_seed(17)
            for shape, lengths in SHAPES.items():
                q, k, v = [
                    torch.randn(
                        sum(lengths),
                        h,
                        256,
                        device="cuda",
                        dtype=torch.bfloat16,
                        requires_grad=True,
                    )
                    for h in (16, 4, 4)
                ]
                cuts = torch.tensor(
                    (0, *accumulate(lengths)), device="cuda", dtype=torch.int32
                )
                grad = torch.randn_like(q)
                kernel = functools.partial(packed_attention, square=True)
                maximum = max(lengths)

                def action(
                    q=q,
                    k=k,
                    v=v,
                    cuts=cuts,
                    maximum=maximum,
                    grad=grad,
                    kernel=kernel,
                ):
                    o = kernel(q, k, v, cuts, maximum)
                    derivatives = torch.autograd.grad(o, (q, k, v), grad)
                    return o, derivatives

                for _ in range(6):
                    out, derivatives = action()
                if not bool(torch.isfinite(out).all()) or not all(
                    bool(torch.isfinite(x).all()) for x in derivatives
                ):
                    raise ValueError("nonfinite timing candidate")
                row = {"shape": shape, "timing": timing(action)}
                report["rows"].append(row)
                save()
                print(json.dumps(row), flush=True)
            report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
