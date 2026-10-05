"""Forward-only frozen projection plus FP32 LoRA producer epilogue pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from itertools import accumulate
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from experiments.b200_meta_stack.producer_reference import norm_rope
from experiments.b200_nvidia_mxfp8_varlen.profile_attention import timing
from gleipnir.nvidia_mxfp8_fused_quantize import prepare
from gleipnir.nvidia_mxfp8_projection_pilot import project_prepare, quantize_weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mxfp8", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "starting",
        "forward_only": True,
        "projection_precision": "mxfp8" if args.mxfp8 else "bf16",
        "rows": [],
    }

    def save():
        (args.output / "projection.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )

    save()
    try:
        config = AutoConfig.from_pretrained(
            "Qwen/Qwen3.5-4B",
            revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            local_files_only=True,
        ).text_config
        inner = config.hidden_size
        heads = config.num_attention_heads
        rotary = int(config.head_dim * config.rope_parameters["partial_rotary_factor"])
        report.update(hidden_size=inner, heads=heads, rotary_dim=rotary)
        torch.backends.cuda.preferred_blas_library("cublaslt")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            False,
            False,
        )
        torch.manual_seed(41)
        # Q-only projection with the real interleaved Q|gate weight strides.
        weight = torch.randn(
            heads * 512, inner, device="cuda", dtype=torch.bfloat16
        ) / math.sqrt(inner)
        selected = (
            weight.view(heads, 512, inner)[:, :256]
            .reshape(heads * 256, inner)
            .contiguous()
        )
        qa, qb = [
            torch.randn(*shape, device="cuda", dtype=torch.float32) * 0.01
            for shape in [(128, inner), (heads * 256, 128)]
        ]
        nw = torch.randn(256, device="cuda", dtype=torch.bfloat16) * 0.1
        payload, sf = quantize_weight(weight) if args.mxfp8 else (weight, None)
        report["weight_preparation_excluded_once"] = args.mxfp8
        for lengths in ((31, 33, 129), (4096,)):
            total = sum(lengths)
            cuts = torch.tensor(
                (0, *accumulate(lengths)), device="cuda", dtype=torch.int32
            )
            hidden = torch.randn(total, inner, device="cuda", dtype=torch.bfloat16)
            angles = torch.randn(total, rotary // 2, device="cuda")
            cos = angles.cos().repeat(1, 2).bfloat16()
            sin = angles.sin().repeat(1, 2).bfloat16()

            # Adapter matmuls remain FP32; their cost is included in both legs.
            def update(hidden=hidden, total=total):
                return (
                    (F.linear(F.linear(hidden.float(), qa), qb) * 2.0)
                    .reshape(total, heads, 256)
                    .contiguous()
                )

            def baseline(
                hidden=hidden, cuts=cuts, total=total, cos=cos, sin=sin, lengths=lengths
            ):
                q = (
                    F.linear(hidden, selected).float().reshape(total, heads, 256)
                    + update()
                ).bfloat16()
                return prepare(
                    norm_rope(q, nw, cos, sin), cuts, max(lengths), square=True
                )

            def candidate(hidden=hidden, cuts=cuts, cos=cos, sin=sin, lengths=lengths):
                return project_prepare(
                    hidden,
                    payload,
                    cuts,
                    max(lengths),
                    heads=heads,
                    head_stride=512,
                    update=update(),
                    norm_weight=nw,
                    cos=cos,
                    sin=sin,
                    weight_sf=sf,
                )

            old = baseline()
            new = candidate()
            agreement = float(
                (old[0].view(torch.uint8) == new[0].view(torch.uint8)).float().mean()
            )
            finite = bool(torch.isfinite(new[0].float()).all())
            row = {"lengths": lengths, "code_agreement": agreement, "finite": finite}
            if args.mxfp8:
                # Arithmetic correctness uses the actual quantized GEMM operands.
                # Difference from unquantized BF16 is recorded separately.
                hx, hs = quantize_weight(hidden)
                dx = (
                    hx.float().reshape(total, inner // 32, 32)
                    * (2.0 ** (hs.float() - 127))[..., None]
                ).reshape(hidden.shape)
                dw = (
                    payload.float().reshape(weight.shape[0], inner // 32, 32)
                    * (2.0 ** (sf.float() - 127))[..., None]
                ).reshape(weight.shape)
                selected_dw = dw.view(heads, 512, inner)[:, :256].reshape(
                    heads * 256, inner
                )
                projected = (
                    F.linear(dx, selected_dw)
                    .bfloat16()
                    .float()
                    .reshape(total, heads, 256)
                    + update()
                ).bfloat16()
                oracle = prepare(
                    norm_rope(projected, nw, cos, sin), cuts, max(lengths), square=True
                )
                oracle_agreement = float(
                    (oracle[0].view(torch.uint8) == new[0].view(torch.uint8))
                    .float()
                    .mean()
                )
                row["quantized_operand_oracle_code_agreement"] = oracle_agreement
            else:
                oracle_agreement = agreement
            if not finite or oracle_agreement < 0.995:
                row["status"] = "rejected_arithmetic"
                report["rows"].append(row)
                save()
                raise ValueError("projection pilot failed arithmetic/finite gate")
            for fn in (baseline, candidate):
                for _ in range(6):
                    fn()
            row.update(
                status="complete",
                baseline=timing(baseline),
                candidate=timing(candidate),
            )
            report["rows"].append(row)
            save()
            print(json.dumps(row), flush=True)
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise
    finally:
        paths = [
            Path(__file__),
            Path("src/gleipnir/nvidia_mxfp8_fused_quantize.py"),
            Path("src/gleipnir/nvidia_mxfp8_projection_pilot.py"),
        ]
        report["source_sha256"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        }
        for p in paths:
            dest = args.output / "executed_source" / p
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
        save()


if __name__ == "__main__":
    main()
